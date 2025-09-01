# simserver/fl/training.py
import os, asyncio, traceback
from typing import Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor, Future
from torch.utils.data import DataLoader
import torch, torch.nn as nn, torch.optim as optim
from torchvision import transforms as _T
import threading

from ..core.logging import make_logger
from ..core.paths import CKPT_DIR
from ..core.gpu import build_worker_gpu_list
from ..core.utils import get_env_int, get_env_float
from .model import new_model_skeleton
from .metrics import log_local_metrics
from ..dataio.registry import (
    get_training_dataset, get_validation_dataset, get_test_dataset,
    DATA_REGISTRY, CIFAR_ROOT
)
from ..sim.state import AppState
from .aggregate import parse_fromg, upload_and_aggregate, get_global_model_snapshot

logger = make_logger("simserver.train")

from torch.cuda.amp import GradScaler

def _dataset_seems_normalized(ds, *,
                              tol_min=-0.05, tol_max=1.05) -> bool:
    """
    텐서 샘플 1개를 확인:
    - 값 범위가 [0,1]에 가깝다면 → 아직 정규화 전(= Normalize 필요) → False
    - 범위를 벗어나면(음수/1 초과 등) → 이미 표준화된 것으로 간주 → True
    """
    try:
        x, _ = ds[0]
        if torch.is_tensor(x):
            m = float(x.min())
            M = float(x.max())
            # 0..1 범위면 아직 Normalize 전
            if (m >= tol_min) and (M <= tol_max):
                return False
            return True
    except Exception:
        pass
    return False

def _maybe_wrap_with_transform(dataset, build_transform_fn):
    """이미 정규화된 텐서를 내는 데이터셋은 건드리지 않음."""
    if _dataset_seems_normalized(dataset):
        return dataset, False
    # torchvision 계열이면 transform 속성 우선 주입
    if hasattr(dataset, "transform"):
        try:
            dataset.transform = build_transform_fn()
            return dataset, True
        except Exception:
            pass
    # 그 외에는 래퍼 사용
    from torchvision import transforms as _T
    class _Wrap(torch.utils.data.Dataset):
        def __init__(self, base, tfm):
            self.base, self.tfm = base, tfm
        def __len__(self): return len(self.base)
        def __getitem__(self, i):
            x, y = self.base[i]
            # x가 텐서면 Normalize만 적용, PIL/ndarray면 Resize+ToTensor+Normalize 전체 적용
            tfm = self.tfm
            if torch.is_tensor(x):
                # 채널순서(CHW) 텐서로 가정: ToTensor 생략, Normalize만
                xmin = float(x.min()); xmax = float(x.max())
                if (xmin >= -1e-3) and (xmax <= 1.0 + 1e-3):
                    mean = tuple(map(float, os.getenv("FL_NORM_MEAN","0.4915,0.4823,0.4468").split(",")))
                    std  = tuple(map(float, os.getenv("FL_NORM_STD" ,"0.2470,0.2435,0.2616").split(",")))
                    mean_t = torch.tensor(mean, dtype=x.dtype, device=x.device)[:, None, None]
                    std_t  = torch.tensor(std,  dtype=x.dtype, device=x.device)[:, None, None]
                    x = (x - mean_t) / std_t
                # 이미 정규화된 값 범위면 그대로 리턴(추가 Normalize 금지)
                return x, y
            else:
                return tfm(x), y
    try:
        from torchvision import transforms as _T
        img_size = int(os.getenv("FL_IMG_SIZE", "32"))
        mean = tuple(map(float, os.getenv("FL_NORM_MEAN","0.4915,0.4823,0.4468").split(",")))
        std  = tuple(map(float, os.getenv("FL_NORM_STD" ,"0.2470,0.2435,0.2616").split(",")))
        tfm = _T.Compose([_T.Resize(img_size), _T.ToTensor(), _T.Normalize(mean, std)])
        return _Wrap(dataset, tfm), True
    except Exception:
        return dataset, False

def _build_train_transform():
    img_size = get_env_int("FL_IMG_SIZE", 32)
    mean = tuple(map(float, os.getenv("FL_NORM_MEAN", "0.4914,0.4822,0.4465").split(",")))
    std  = tuple(map(float, os.getenv("FL_NORM_STD",  "0.2470,0.2435,0.2616").split(",")))
    # 필요하면 augmentation을 여기 추가 (RandomCrop/Flip 등)
    return _T.Compose([
        _T.Resize(img_size),
        _T.ToTensor(),
        _T.Normalize(mean, std),
    ])

def _global_grad_norm(model) -> float | None:
    """Return L2 norm of all gradients if available, else None."""
    sqsum = 0.0
    n = 0
    for p in model.parameters():
        if p.grad is not None:
            g = p.grad.detach()
            sqsum += float(g.pow(2).sum().item())
            n += 1
    if n == 0:
        return None
    return float(sqsum) ** 0.5

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

def local_train(ctx: AppState, sat_id: int, stop_event: threading.Event, gpu_id: Optional[int] = None,
                      *, epochs=None, lr=None, batch_size=None) -> str:
    EPOCHS = get_env_int("FL_EPOCHS_PER_ROUND", 1) if epochs is None else int(epochs)
    LR     = get_env_float("FL_LR", 1e-2) if lr is None else float(lr)
    BS     = get_env_int("FL_BATCH_SIZE", 64) if batch_size is None else int(batch_size)
    NUM_CLASSES = ctx.cfg.num_classes

    USE_AMP = os.getenv("FL_USE_AMP", "1") == "1"
    MAX_BAD_SKIPS = get_env_int("FL_MAX_BAD_BATCH_SKIPS", 20)
    MAX_STALENESS = get_env_int("MAX_STALENESS", get_env_int("FL_MAX_STALENESS", 6))

    device = torch.device(f"cuda:{gpu_id}") if (torch.cuda.is_available() and gpu_id is not None) else (
             torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    init_scale_env = os.getenv("FL_AMP_INIT_SCALE", "65536")  # 선택: 환경에서 받기
    try:
        init_scale = float(init_scale_env) if init_scale_env is not None else 65536.0
    except Exception:
        logger.warning(f"SAT{sat_id}: invalid FL_AMP_INIT_SCALE='{init_scale_env}', disable AMP for safety.")
        USE_AMP = False
        init_scale = 1.0

    scaler = GradScaler(enabled=(USE_AMP and device.type == "cuda"), init_scale=init_scale)

    model = new_model_skeleton(NUM_CLASSES)
    try:
        last_linear = None
        for m in model.modules():
            if isinstance(m, nn.Linear):
                last_linear = m
        if last_linear is not None and last_linear.out_features != NUM_CLASSES:
            logger.warning(f"[SANITY] head out_features={last_linear.out_features} != NUM_CLASSES={NUM_CLASSES}")
    except Exception:
        pass

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
            if (ctx.GLOBAL_MODEL_VERSION - start_gver) > max(0, MAX_STALENESS):
                logger.info(f"SAT{sat_id}: resume too stale (from_gv={start_gver}, cur={ctx.GLOBAL_MODEL_VERSION}); reinit from current global.")
                gver, gstate = get_global_model_snapshot(ctx)
                if gstate:
                    model.load_state_dict(gstate, strict=False)
                    start_gver = int(gver); resumed = False
            else:
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

    MU = float(os.getenv("FL_FEDPROX_MU", "0"))
    global_params = None
    if MU > 0:
        global_params = {name: p.detach().clone().to(device) for name, p in model.named_parameters()}

    ENFORCE_NORM = os.getenv("FL_ENFORCE_TRAIN_NORM", "1") == "1"
    if ENFORCE_NORM:
        dataset, applied = _maybe_wrap_with_transform(dataset, _build_train_transform)
        logger.info(f"SAT{sat_id}: train normalization {'applied' if applied else 'skipped (already normalized)'}")



    pin = torch.cuda.is_available() and ctx.cfg.dataloader_workers > 0
    loader = DataLoader(dataset, batch_size=BS, shuffle=True, drop_last=False,
                        num_workers=ctx.cfg.dataloader_workers, pin_memory=pin,
                        persistent_workers=(ctx.cfg.dataloader_workers>0))

    criterion = nn.CrossEntropyLoss()
    WD    = get_env_float("FL_WEIGHT_DECAY", 5e-4)  # ★ L2 정규화(권장: CIFAR 5e-4)
    MOM   = get_env_float("FL_MOMENTUM", 0.9)
    optimizer = optim.SGD(
            model.parameters(), lr=LR, momentum=MOM, weight_decay=WD,
            nesterov=(os.getenv("FL_NESTEROV","1")=="1")
    )         
    clip_norm = float(os.getenv("FL_CLIP_NORM","1.0"))


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

        did_sanity_log = False

        for images, labels in loader:
            if stop_event.is_set(): break

            if labels.dtype != torch.long:
                labels = labels.long()
            
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if not did_sanity_log:
                with torch.no_grad():
                    m = float(images.mean().detach().cpu())
                    s = float(images.std().detach().cpu())
                    y_min = int(labels.min().detach().cpu()) if labels.numel() else -1
                    y_max = int(labels.max().detach().cpu()) if labels.numel() else -1
                    logger.info(f"SAT{sat_id}: sanity img mean={m:.3f} std={s:.3f} "
                                f"label_range=[{y_min},{y_max}] C={NUM_CLASSES}")
                did_sanity_log = True

            optimizer.zero_grad(set_to_none=True)
            try:
                if scaler.is_enabled() and device.type == "cuda":
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                else:
                    outputs = model(images)
                    loss = criterion(outputs, labels)
            except RuntimeError as e:
                msg = str(e)
                if ("Expected all tensors to be on the same device" in msg) or ("and weight type" in msg):
                    device = ensure_all_on_model_device(model, optimizer, criterion)
                    images = images.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                else:
                    raise

            if MU > 0 and global_params is not None:
                prox = 0.0
                for name, param in model.named_parameters():
                    prox += (param - global_params[name]).pow(2).sum()
                loss = loss + (MU / 2.0) * prox

            if not torch.isfinite(loss):
                bad_skips += 1
                try:
                    img_m, img_s = float(images.mean().item()), float(images.std().item())
                    img_min, img_max = float(images.min().item()), float(images.max().item())
                except Exception:
                    img_m = img_s = img_min = img_max = float("nan")

                try:
                    # outputs may contain NaN; use nan-aware stats
                    out_m = float(torch.nanmean(outputs.float()).item())
                    out_absmax = float(torch.nanmax(outputs.float().abs()).item())
                except Exception:
                    out_m = out_absmax = float("nan")

                try:
                    lr_now = float(optimizer.param_groups[0]["lr"])
                except Exception:
                    lr_now = float("nan")

                try:
                    gnorm = _global_grad_norm(model)  # may be None if no grads yet
                except Exception:
                    gnorm = None

                logger.warning(
                    f"SAT{sat_id}: non-finite loss detected | "
                    f"img(mean/std/min/max)={img_m:.4g}/{img_s:.4g}/{img_min:.4g}/{img_max:.4g} "
                    f"out(mean/|max|)={out_m:.4g}/{out_absmax:.4g} lr={lr_now:.2e} grad_norm={gnorm}"
                )

                if bad_skips % 5 == 0:
                    for g in optimizer.param_groups:
                        g['lr'] = max(g['lr'] * 0.5, 1e-6)
                    logger.info(f"SAT{sat_id}: non-finite loss; halve LR -> {optimizer.param_groups[0]['lr']:.2e}")
                if bad_skips >= MAX_BAD_SKIPS:
                    logger.info(f"SAT{sat_id}: too many bad batches; break epoch (AMP={USE_AMP})")
                    break
                continue


            # backward + step (AMP/FP32 공통화)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                if clip_norm > 0:
                    scaler.unscale_(optimizer)  # 클립 전에 unscale (중요)
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if clip_norm > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
                optimizer.step()

            running_loss += float(loss.item())
            _, pred = outputs.max(1)
            total += labels.size(0)
            correct += int(pred.eq(labels).sum().item())

        if total == 0 and scaler.is_enabled():
            logger.warning(f"SAT{sat_id}: epoch{ep+1} produced no valid steps; disabling AMP for stability.")
            scaler = GradScaler(enabled=False)

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
                return local_train(ctx, sat_id=sat_id, stop_event=st.stop_event, gpu_id=gpu_id)

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

                # ★★★ OFFLINE 모드: 통신 불가해도 로컬 학습 계속 돌리기 ★★★
                maybe_reenqueue_offline(ctx, sat_id)
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
        logger.warning(f"train queue full; drop sat{sat_id}")

def maybe_reenqueue_offline(ctx: AppState, sat_id: int):
    if os.getenv("FL_OFFLINE_CONTINUE","1") != "1":
        return
    try:
        enqueue_training(ctx, sat_id)
        logger.info(f"SAT{sat_id}: re-enqueued for offline-continue.")
    except Exception as e:
        logger.warning(f"SAT{sat_id}: offline-continue enqueue failed: {e}")


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
                # --- NOTE/INFO: n_samples를 집계에 전달하여 alpha 스케일 기대/관측을 정렬 ---
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
