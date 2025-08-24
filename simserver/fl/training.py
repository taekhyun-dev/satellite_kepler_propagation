# simserver/fl/training.py
import os, asyncio, time, traceback
from typing import Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor, Future
from torch.utils.data import DataLoader
import torch, torch.nn as nn, torch.optim as optim
import threading

from ..core.logging import make_logger
from ..core.paths import CKPT_DIR
from ..core.gpu import build_worker_gpu_list
from ..core.utils import get_env_int
from .model import new_model_skeleton
from .metrics import log_local_metrics
from ..dataio.registry import (
    get_training_dataset, get_validation_dataset, get_test_dataset,
    DATA_REGISTRY, CIFAR_ROOT, ASSIGNMENTS_DIR,
    SAMPLES_PER_CLIENT, DIRICHLET_ALPHA, RNG_SEED, WITH_REPLACEMENT
)
from ..sim.skyfiled import to_ts
from ..sim.state import AppState, TrainState
from .aggregate import parse_fromg, upload_and_aggregate, get_global_model_snapshot

logger = make_logger("simserver.train")

def ensure_all_on_model_device(model, optimizer, criterion=None):
    dev = next(model.parameters()).device
    model.to(dev)
    if criterion is not None:
        if hasattr(criterion, 'weight') and torch.is_tensor(criterion.weight):
            criterion.weight = criterion.weight.to(dev, non_blocking=True)
        try: criterion.to(dev)
        except Exception: pass
    if optimizer is not None:
        for state in optimizer.state.values():
            for k, v in list(state.items()):
                if torch.is_tensor(v):
                    state[k] = v.to(dev, non_blocking=True)
    return dev

def local_ptr_path(sat_id: int):
    return CKPT_DIR / f"sat{sat_id}_last.json"

def save_last_local_ptr(sat_id: int, path: str, *, from_gver=None, round_idx=None, epoch=None):
    import json
    p = local_ptr_path(sat_id)
    tmp = p.with_suffix(".json.tmp")
    meta = {
        "path": str(path),
        "from_gver": int(from_gver) if from_gver is not None else (parse_fromg(os.path.basename(path)) or -1),
        "round_idx": int(round_idx) if round_idx is not None else None,
        "epoch": int(epoch) if epoch is not None else None,
        "updated_at": __import__("datetime").datetime.now().isoformat(),
    }
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(meta, f)
    tmp.replace(p)

def load_last_local_ptr(sat_id: int) -> Optional[Dict[str, Any]]:
    import json
    p = local_ptr_path(sat_id)
    if not p.exists(): return None
    with p.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    path = meta.get("path")
    if path and os.path.exists(path):
        return meta
    return None

def get_eval_dataset(ctx: AppState, split: str):
    if ctx.EVAL_DS_CACHE.get(split) is not None:
        return ctx.EVAL_DS_CACHE[split]
    # 1) DATA_REGISTRY methods
    try:
        meth_name = f"get_{'validation' if split=='val' else 'test'}_dataset"
        meth = getattr(DATA_REGISTRY, meth_name, None)
        if callable(meth):
            ds = meth()
            if ds is not None:
                ctx.EVAL_DS_CACHE[split] = ds
                return ds
    except Exception as e:
        logger.warning(f"[EVAL] DATA_REGISTRY.{split} failed: {e}")
    # 2) data module functions
    try:
        ds = get_validation_dataset() if split=="val" else get_test_dataset()
        ctx.EVAL_DS_CACHE[split] = ds
        return ds
    except Exception as e:
        logger.warning(f"[EVAL] data.{split} fallback failed: {e}")
    # 3) torchvision CIFAR10(test)
    try:
        from torchvision.datasets import CIFAR10
        from torchvision import transforms
        img_size = get_env_int("FL_IMG_SIZE", 32)
        mean = tuple(map(float, os.getenv("FL_NORM_MEAN","0.4914,0.4822,0.4465").split(",")))
        std  = tuple(map(float, os.getenv("FL_NORM_STD" ,"0.2470,0.2435,0.2616").split(",")))
        tfm = transforms.Compose([transforms.Resize(img_size), transforms.ToTensor(), transforms.Normalize(mean, std)])
        root_dir = str(CIFAR_ROOT) if 'CIFAR_ROOT' in globals() else str((CKPT_DIR.parent / "data"))
        ds = CIFAR10(root=root_dir, train=False, download=True, transform=tfm)
        ctx.EVAL_DS_CACHE[split] = ds
        if "fallback" not in getattr(ctx, "_EVAL_FALLBACK_LOGGED", set()):
            logger.info(f"[EVAL] fallback CIFAR10(test) | size={img_size}, norm=mean{mean},std{std}")
        return ds
    except Exception as e:
        logger.warning(f"[EVAL] torchvision CIFAR10 fallback failed: {e}")
        return None

def do_local_training(ctx: AppState, sat_id: int, stop_event: threading.Event, gpu_id: Optional[int] = None,
                      *, epochs=None, lr=None, batch_size=None) -> str:
    EPOCHS = int(os.getenv("FL_EPOCHS_PER_ROUND","2")) if epochs is None else int(epochs)
    LR     = float(os.getenv("FL_LR","0.05"))           if lr is None else float(lr)
    BS     = int(os.getenv("FL_BATCH_SIZE","128"))      if batch_size is None else int(batch_size)
    NUM_CLASSES = ctx.cfg.num_classes

    device = torch.device(f"cuda:{gpu_id}") if (torch.cuda.is_available() and gpu_id is not None) else (
             torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    model = new_model_skeleton(NUM_CLASSES)
    start_gver = ctx.GLOBAL_MODEL_VERSION

    resumed = False
    try:
        meta = load_last_local_ptr(sat_id)
        if meta and meta.get("path"):
            sd = torch.load(meta["path"], map_location="cpu")
            model.load_state_dict(sd, strict=False)
            fg = meta.get("from_gver")
            if fg is None or fg < 0:
                fg = parse_fromg(os.path.basename(meta["path"])) or start_gver
            start_gver = int(fg); resumed = True
            logger.info(f"SAT{sat_id}: resumed from {meta['path']} (from_gv={start_gver})")
    except Exception as e:
        logger.info(f"SAT{sat_id}: no local resume; use global. err={e}")

    # resume를 못했으면 글로벌 스냅샷에서 시작
    if not resumed:
        try:
            gver, gstate = get_global_model_snapshot(ctx)
            if gstate:
                model.load_state_dict(gstate, strict=False)
                start_gver = int(gver)
                logger.info(f"SAT{sat_id}: init from global v{gver}")
        except Exception as e:
            logger.warning(f"SAT{sat_id}: failed to load global snapshot: {e}")

    model.to(device).train()
    dataset = get_training_dataset(sat_id)
    pin = torch.cuda.is_available() and ctx.cfg.dataloader_workers > 0
    loader = DataLoader(dataset, batch_size=BS, shuffle=True, drop_last=False,
                        num_workers=ctx.cfg.dataloader_workers, pin_memory=pin,
                        persistent_workers=(ctx.cfg.dataloader_workers>0))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LR)
    clip_norm = float(os.getenv("FL_CLIP_NORM","1.0"))
    max_bad_skips = int(os.getenv("FL_MAX_BAD_BATCH_SKIPS","20"))

    def save_ckpt(ep: int) -> str:
        ts = __import__("datetime").datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"sat{sat_id}_fromg{start_gver}_round{ctx.train_states[sat_id].round_idx}_ep{ep}_{ts}.ckpt"
        ckpt_path = CKPT_DIR / fname
        torch.save(model.state_dict(), ckpt_path)
        save_last_local_ptr(sat_id, str(ckpt_path), from_gver=start_gver,
                            round_idx=ctx.train_states[sat_id].round_idx, epoch=ep)
        try: ctx.train_states[sat_id].last_ckpt_path = str(ckpt_path)
        except Exception: pass
        return str(ckpt_path)

    last_ckpt, n_total = None, (len(dataset) if hasattr(dataset, "__len__") else None)

    for ep in range(EPOCHS):
        if stop_event.is_set(): break
        running_loss, correct, total = 0.0, 0, 0
        bad_skips = 0
        device = ensure_all_on_model_device(model, optimizer, criterion)
        model_dtype = next(model.parameters()).dtype

        for images, labels in loader:
            if stop_event.is_set(): break
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if images.is_floating_point() and images.dtype != model_dtype:
                images = images.to(model_dtype)

            optimizer.zero_grad(set_to_none=True)
            use_amp = (device.type == "cuda")
            try:
                if use_amp:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        outputs = model(images)
                else:
                    outputs = model(images)
            except RuntimeError as e:
                msg = str(e)
                if ("Expected all tensors to be on the same device" in msg) or ("and weight type" in msg):
                    device = ensure_all_on_model_device(model, optimizer, criterion)
                    images = images.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    outputs = model(images)
                else:
                    raise

            loss = criterion(outputs, labels)
            if not torch.isfinite(loss):
                bad_skips += 1
                if bad_skips % 5 == 0:
                    for g in optimizer.param_groups:
                        g['lr'] = max(g['lr'] * 0.5, 1e-6)
                    logger.info(f"SAT{sat_id}: non-finite loss; halve LR -> {optimizer.param_groups[0]['lr']:.2e}")
                if bad_skips >= max_bad_skips:
                    logger.info(f"SAT{sat_id}: too many bad batches; break epoch")
                    break
                continue

            loss.backward()
            if clip_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
            optimizer.step()

            running_loss += float(loss.item())
            _, pred = outputs.max(1)
            total += labels.size(0)
            correct += int(pred.eq(labels).sum().item())

        avg_loss = running_loss / max(1, len(loader))
        acc = 100.0 * correct / max(1, total)
        last_ckpt = save_ckpt(ep)
        logger.info(f"SAT{sat_id}: ep={ep+1}/{EPOCHS} loss={avg_loss:.4f} acc={acc:.2f}% from_gv={start_gver} -> {last_ckpt}")
        try:
            log_local_metrics(sat_id, ctx.train_states[sat_id].round_idx, ep,
                              avg_loss, acc, n_total or total, last_ckpt)
        except Exception as e:
            logger.warning(f"[METRICS] local write failed (sat{sat_id}): {e}")

    return last_ckpt or save_ckpt(-1)

# --- Worker orchestration ---

async def train_worker(ctx: AppState, gpu_id: Optional[int]):
    try:
        loop = asyncio.get_running_loop()
        while True:
            sat_id = await ctx.train_queue.get()
            st = ctx.train_states[sat_id]
            st.in_queue = False
            if st.running:
                ctx.train_queue.task_done()
                continue
            st.stop_event.clear()
            st.gpu_id = gpu_id
            st.running = True

            def _job():
                return do_local_training(ctx, sat_id=sat_id, stop_event=st.stop_event, gpu_id=gpu_id)

            fut = loop.run_in_executor(ctx.training_executor, _job)
            st.future = fut
            try:
                ckpt = await fut
            except Exception as e:
                tb = traceback.format_exc()
                logger.error(f"SAT{sat_id}: training ERROR: {e}\n{tb}")
                ckpt = None
            finally:
                st.last_ckpt_path = ckpt
                st.running = False
                logger.info(f"SAT{sat_id}: training DONE (gpu={gpu_id}), ckpt={ckpt}")
                ctx.train_queue.task_done()
    except asyncio.CancelledError:
        logger.info("[TRAIN] worker cancelled")
        raise

def enqueue_training(ctx: AppState, sat_id: int):
    st = ctx.train_states[sat_id]
    if st.running or st.in_queue: return
    try:
        ctx.train_queue.put_nowait(sat_id)
        st.in_queue = True
    except asyncio.QueueFull:
        pass

def on_become_visible(ctx: AppState, sat_id: int):
    st = ctx.train_states[sat_id]
    def _upload_job():
        ck = ctx.train_states[sat_id].last_ckpt_path
        if ck:
            try:
                n = None
                try:
                    n = len(get_training_dataset(sat_id))
                except Exception:
                    pass
                new_global = upload_and_aggregate(ctx, sat_id, ck, n_samples=n)   # 무거운 작업: 별도 스레드에서 수행
                logger.info(f"SAT{sat_id}: aggregated -> {new_global}")
            except Exception as e:
                logger.error(f"SAT{sat_id}: upload/aggregate ERROR: {e}")

    if st.running and st.future:
        st.stop_event.set()
        def _cb(_fut: Future):
            try:
                _fut.result()   # 예외 소거
            except Exception:
                pass
            # ★ 업로드/집계는 업로더 풀에서 실행
            ctx.uploader_executor.submit(_upload_job)
        try:
            st.future.add_done_callback(_cb)
        except Exception:
            # 일부 런타임에서 add_done_callback가 타입 이슈나면 우회
            def _safe():
                try:
                    _cb(st.future)
                except Exception:
                    pass
            ctx.training_executor.submit(_safe)
    else:
        ctx.uploader_executor.submit(_upload_job)

def start_executors_and_workers(ctx: AppState):
    ctx.training_executor = ThreadPoolExecutor(max_workers=ctx.cfg.max_train_workers)
    ctx.uploader_executor = ThreadPoolExecutor(max_workers=4)
    ctx.train_queue = asyncio.Queue(maxsize=int(os.getenv("FL_QUEUE_MAX","1000")))
    for gid in build_worker_gpu_list(ctx.cfg.num_gpus, ctx.cfg.sessions_per_gpu):
        asyncio.create_task(train_worker(ctx, gid))
