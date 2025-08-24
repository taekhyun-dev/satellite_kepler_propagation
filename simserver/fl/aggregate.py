# simserver/fl/aggregate.py
import os, re, json, math
import torch
from pathlib import Path
from typing import Optional
from ..core.paths import CKPT_DIR, LAST_GLOBAL_PTR
from ..core.logging import make_logger
from .eval import evaluate_state_dict, bn_recalibrate
from .metrics import log_global_metrics
import copy
from ..sim.state import AppState
from ..dataio.registry import get_training_dataset

logger = make_logger("simserver.agg")

def _calc_norm(d, keys=None):
    tot = 0.0
    if keys is None: keys = d.keys()
    for k in keys:
        t = d[k]
        if not torch.is_tensor(t): continue
        t = t.float().view(-1)
        tot += float(torch.dot(t, t).cpu())
    return math.sqrt(max(tot, 0.0))

def get_global_model_snapshot(ctx: AppState):
    """(version, deepcopy(state_dict))를 원자적으로 반환"""
    with ctx.GLOBAL_MODEL_LOCK:
        ver = ctx.GLOBAL_MODEL_VERSION
        state = copy.deepcopy(ctx.GLOBAL_MODEL_STATE)
    return ver, state

def parse_fromg(ckpt_path: str) -> Optional[int]:
    m = re.search(r"fromg(\d+)", os.path.basename(ckpt_path))
    return int(m.group(1)) if m else None

def staleness_factor(s: int, tau: float, mode: str) -> float:
    import math
    if tau <= 0: return 1.0
    if mode == "poly": return (1.0 + s) ** (-tau)
    return math.exp(-float(s) / tau)

def alpha_for(ctx: AppState, sat_id: int, s: int, n_samples_override: Optional[int] = None) -> float:
    if s >= max(ctx.cfg.fresh_cutoff, 0):
        return 0.0

    decay = staleness_factor(s, ctx.cfg.staleness_tau, ctx.cfg.staleness_mode)

    def _client_num_samples(_sid: int) -> int:
        # 우선 override
        if (n_samples_override is not None) and (_sid == sat_id):
            return int(max(1, n_samples_override))
        try:
            ds = get_training_dataset(_sid)
            return len(ds) if hasattr(ds, "__len__") else 1
        except Exception:
            return 1

    # 평균 샘플 수
    try:
        mean_n = sum(_client_num_samples(sid) for sid in ctx.satellites) / max(1, len(ctx.satellites))
    except Exception:
        mean_n = _client_num_samples(sat_id)

    scale = _client_num_samples(sat_id) / max(1e-9, mean_n)
    alpha_eff = float(ctx.cfg.agg_alpha * decay * scale)

    if ctx.cfg.w_min > 0.0: alpha_eff = max(ctx.cfg.w_min, alpha_eff)
    alpha_eff = min(alpha_eff, ctx.cfg.alpha_max)
    return float(alpha_eff)

def aggregate_params(global_state: dict, local_state: dict, alpha: float, bn_scale: float) -> dict:
    import torch
    out = {}
    for k, g_t in global_state.items():
        l_t = local_state.get(k)
        if l_t is None or getattr(g_t, "shape", None) != getattr(l_t, "shape", None):
            out[k] = g_t.clone() if hasattr(g_t, "clone") else g_t
            continue
        if k.endswith("running_mean") or k.endswith("running_var"):
            if hasattr(g_t, "is_floating_point") and g_t.is_floating_point() and l_t.is_floating_point():
                g = g_t.detach().to("cpu", dtype=torch.float32)
                l = l_t.detach().to("cpu", dtype=torch.float32)
                a = float(alpha) * float(bn_scale)
                out[k] = ((1.0 - a) * g + a * l).to(dtype=g_t.dtype)
            else:
                out[k] = g_t.clone()
            continue
        if k.endswith("num_batches_tracked"):
            out[k] = g_t.clone()
            continue
        if hasattr(g_t, "is_floating_point") and g_t.is_floating_point() and l_t.is_floating_point():
            g = g_t.detach().to("cpu", dtype=torch.float32)
            l = l_t.detach().to("cpu", dtype=torch.float32)
            out[k] = ((1.0 - alpha) * g + alpha * l).to(dtype=g_t.dtype)
        else:
            out[k] = g_t.clone() if hasattr(g_t, "clone") else g_t
    return out

def save_last_global_ptr(path: Path, version: int):
    try:
        tmp = LAST_GLOBAL_PTR.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump({"path": str(path), "version": int(version)}, f)
        tmp.replace(LAST_GLOBAL_PTR)
    except Exception as e:
        logger.warning(f"[AGG] write last_global.json failed: {e}")

def load_last_global_ptr() -> tuple[int, Optional[Path]]:
    try:
        if LAST_GLOBAL_PTR.exists():
            with LAST_GLOBAL_PTR.open("r", encoding="utf-8") as f:
                d = json.load(f)
            p = Path(d.get("path", ""))
            v = int(d.get("version", -1))
            if p.exists():
                return v, p
    except Exception as e:
        logger.warning(f"[AGG] read last_global.json failed: {e}")
    return -1, None

def find_latest_global_ckpt():
    import re
    paths = list(CKPT_DIR.glob("global_v*.ckpt"))
    best_ver, best_ts, best_path = -1, "", None
    for p in paths:
        m = re.match(r"global_v(\d+)(?:_(\d{8}_\d{6}))?\.ckpt$", p.name)
        if not m: continue
        v = int(m.group(1)); ts = m.group(2) or ""
        if (v > best_ver) or (v == best_ver and ts > best_ts):
            best_ver, best_ts, best_path = v, ts, p
    return best_ver, best_path

def init_global_model(ctx: AppState):
    import torch
    from .model import new_model_skeleton

    with ctx.GLOBAL_MODEL_LOCK:
        ver, path = load_last_global_ptr()
        if path is None:
            ver, path = find_latest_global_ckpt()

        if path and path.exists() and ver >= 0:
            try:
                ctx.GLOBAL_MODEL_STATE = torch.load(path, map_location="cpu")
                ctx.GLOBAL_MODEL_VERSION = ver
                logger.info(f"[AGG] Loaded global model v{ctx.GLOBAL_MODEL_VERSION} from {path}")
                return
            except Exception as e:
                logger.warning(f"[AGG] load fail ({path}): {e}. Re-initializing...")

        model = new_model_skeleton(ctx.cfg.num_classes)
        ctx.GLOBAL_MODEL_STATE = model.state_dict()
        ctx.GLOBAL_MODEL_VERSION = 0

        ts = __import__("datetime").datetime.now().strftime("%Y%m%d_%H%M%S")
        init_path = CKPT_DIR / f"global_v0_{ts}.ckpt"
        link_path = CKPT_DIR / "global_v0.ckpt"
        try:
            torch.save(ctx.GLOBAL_MODEL_STATE, init_path)
            torch.save(ctx.GLOBAL_MODEL_STATE, link_path)
        except Exception as e:
            logger.warning(f"[AGG] initial save warn: {e}")
        save_last_global_ptr(init_path, 0)
        logger.info(f"[AGG] Initialized new global model at {init_path}")

def upload_and_aggregate(
        ctx: AppState, sat_id: int, ckpt_path: str, *,
        n_samples: Optional[int] = None,   # <-- 호출부 호환용(선택적)
        **_ignored                         # <-- 기타 예기치 않은 키워드도 무시
    ) -> str:
    import os, torch
    from pathlib import Path

    if not ckpt_path or not Path(ckpt_path).exists():
        raise FileNotFoundError(f"ckpt not found: {ckpt_path}")

    local_state = torch.load(ckpt_path, map_location="cpu")
    with ctx.GLOBAL_MODEL_LOCK:
        # --- staleness/alpha 계산
        base_ver = parse_fromg(ckpt_path)
        gver_now = ctx.GLOBAL_MODEL_VERSION
        s = max(0, gver_now - (base_ver if base_ver is not None else gver_now))

        stream_fedavg = os.getenv("FL_STREAM_FEDAVG", "1") == "1"
        # 누적 샘플 상태 초기화
        if stream_fedavg:
            if not hasattr(ctx, "_round_id"):   ctx._round_id = None    # ★
            if not hasattr(ctx, "_round_seen"): ctx._round_seen = 0     # ★
            this_round = (base_ver if base_ver is not None else gver_now)
            if ctx._round_id != this_round:     # 라운드 변경 시 리셋        # ★
                ctx._round_id = this_round
                ctx._round_seen = 0

        if ctx.cfg.s_max_drop > 0 and s > ctx.cfg.s_max_drop and not stream_fedavg:
            logger.info(f"[AGG] drop stale update: s={s} (> {ctx.cfg.s_max_drop}) from {ckpt_path}")
            return str(CKPT_DIR / f"global_v{ctx.GLOBAL_MODEL_VERSION}.ckpt")

        # α 계산
        if stream_fedavg:
            nk = int(n_samples) if (n_samples is not None) else 1       # ★
            alpha_eff = nk / max(1, ctx._round_seen + nk)                # ★ 정확한 가중 평균 계수
        else:
            alpha_eff = alpha_for(ctx, sat_id, s)


        # --- 글로벌이 비어 있거나 첫 초기화인 경우 보호
        if ctx.GLOBAL_MODEL_STATE is None or len(ctx.GLOBAL_MODEL_STATE) == 0:
            ctx.GLOBAL_MODEL_STATE = {k: v.detach().clone().cpu() for k, v in local_state.items()}
            ctx.GLOBAL_MODEL_VERSION = 0
            ts = __import__("datetime").datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = CKPT_DIR / f"global_v{ctx.GLOBAL_MODEL_VERSION}_{ts}.ckpt"
            torch.save(ctx.GLOBAL_MODEL_STATE, out_path)
            try: torch.save(ctx.GLOBAL_MODEL_STATE, CKPT_DIR / f"global_v{ctx.GLOBAL_MODEL_VERSION}.ckpt")
            except Exception: pass
            save_last_global_ptr(out_path, ctx.GLOBAL_MODEL_VERSION)
            logger.info(f"[AGG] warm-start global (copy from sat{sat_id}) -> v{ctx.GLOBAL_MODEL_VERSION}")
            return str(out_path)

        if alpha_eff <= 0.0:
            ts = __import__("datetime").datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = CKPT_DIR / f"global_v{ctx.GLOBAL_MODEL_VERSION}_{ts}.ckpt"
            try: torch.save(ctx.GLOBAL_MODEL_STATE, out_path)
            except Exception: pass
            return str(out_path)

        # --- 키 교집합/미스/BN 스킵 집계
        g_state = ctx.GLOBAL_MODEL_STATE
        g_keys = set(g_state.keys())
        l_keys = set(local_state.keys())
        matched = sorted(list(g_keys & l_keys))
        miss_g = sorted(list(g_keys - l_keys))
        miss_l = sorted(list(l_keys - g_keys))

        # 서버 모멘텀 (FedAvgM)
        beta = float(os.getenv("FL_SERVER_MOM", "0.9"))  # 0.9 권장
        if not hasattr(ctx, "GLOBAL_VEL"):   # velocity buffer
            ctx.GLOBAL_VEL = {}

        bn_scale = float(os.getenv("FL_BN_SCALE", "0.2"))  # BN affine 섞기 축소
        # BN 러닝통계는 절대 섞지 않음
        def _is_bn_stat(k: str) -> bool:
            return (k.endswith("running_mean") or k.endswith("running_var") or ("num_batches_tracked" in k))

        def _is_bn_affine(k: str) -> bool:
            # 레이어 이름에 'bn'이 들어가고 .weight/.bias 인 경우
            return (("bn" in k or ".bn" in k) and (k.endswith(".weight") or k.endswith(".bias")))

        included = [k for k in matched
                    if (not _is_bn_stat(k)) and torch.is_floating_point(g_state[k])]

        # 집계 전/후 변화량(진단용)
        def _flat_norm(d: dict, keys):
            s = 0.0
            for kk in keys:
                t = d[kk]
                if torch.is_floating_point(t):
                    s += float(torch.norm(t.float()).item() ** 2)
            return (s ** 0.5) if s > 0 else 0.0
        
        g_pre = {k: g_state[k].detach().cpu().float().clone() for k in included}
        pre_vec_norm = _flat_norm(g_state, included) + 1e-12  # eps
        theo_sumsq = 0.0

        for k in included:
            g = g_pre[k]
            l = local_state[k].detach().cpu().float()
            diff = (l - g).view(-1)
            a_k = alpha_eff * (bn_scale if _is_bn_affine(k) else 1.0)
            # (a_k * diff) 벡터의 제곱합
            theo_sumsq += float(torch.dot(diff, diff)) * (a_k ** 2)
        theo_rel = math.sqrt(theo_sumsq) / pre_vec_norm

        skipped_bn = 0
        for k in matched:
            if _is_bn_stat(k):
                # BN running_mean/running_var/num_batches_tracked: 건드리지 않음
                skipped_bn += 1
                continue

            g = g_state[k].detach().cpu().float()
            l = local_state[k].detach().cpu().float()
            delta = l - g

            # BN affine은 축소된 alpha로
            alpha_k = (alpha_eff * (bn_scale if _is_bn_affine(k) else 1.0))

            # 모멘텀 업데이트
            v = ctx.GLOBAL_VEL.get(k, torch.zeros_like(delta))
            if beta > 0.0:
                v = v * beta + delta * alpha_k
                g_new = g + v
                ctx.GLOBAL_VEL[k] = v
            else:
                g_new = g + delta * alpha_k

            # dtype/디바이스 원복
            g_state[k] = g_new.to(ctx.GLOBAL_MODEL_STATE[k].dtype)

        if stream_fedavg:
            ctx._round_seen += int(n_samples) if (n_samples is not None) else 1

        applied_sumsq = 0.0
        for k in included:
            g_post = g_state[k].detach().cpu().float().view(-1)
            g_pre_v = g_pre[k].view(-1)
            diff = g_post - g_pre_v
            applied_sumsq += float(torch.dot(diff, diff))
        applied_rel = math.sqrt(applied_sumsq) / pre_vec_norm

        ctx.GLOBAL_MODEL_STATE = g_state
        old_ver = ctx.GLOBAL_MODEL_VERSION
        ctx.GLOBAL_MODEL_VERSION += 1

        ts = __import__("datetime").datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path  = CKPT_DIR / f"global_v{ctx.GLOBAL_MODEL_VERSION}_{ts}.ckpt"
        link_path = CKPT_DIR / f"global_v{ctx.GLOBAL_MODEL_VERSION}.ckpt"
        torch.save(ctx.GLOBAL_MODEL_STATE, out_path)
        try: torch.save(ctx.GLOBAL_MODEL_STATE, link_path)
        except Exception: pass
        save_last_global_ptr(out_path, ctx.GLOBAL_MODEL_VERSION)

        logger.info(
            f"AGG sat={sat_id} g:{old_ver}->{ctx.GLOBAL_MODEL_VERSION} from_g={base_ver} "
            f"α_eff={alpha_eff:.4f} stl={s} match={len(matched)} miss(g,l)=({len(miss_g)},{len(miss_l)}) "
            f"bn_skip={skipped_bn} Δ‖W‖/‖W‖={applied_rel:.3e} (exp_no_mom={theo_rel:.3e}) -> {out_path}"
        )
    # --- (중요) BN 재보정 + 평가
    try:
        # EVAL_EVERY_N=1 권장
        if ctx.GLOBAL_MODEL_VERSION % max(1, ctx.cfg.eval_every_n) == 0:
            from .training import get_eval_dataset
            # 가벼운 sanity-eval을 먼저 수행(재보정 전)
            def _quick_eval(state_dict, ds, max_batches=50):
                if ds is None: return None, None, 0
                from .model import new_model_skeleton
                model = new_model_skeleton(ctx.cfg.num_classes)
                missing, unexpected = model.load_state_dict(state_dict, strict=False)
                if missing or unexpected:
                    logger.warning(f"[AGG] eval load_state mismatch: missing={len(missing)}, unexpected={len(unexpected)}")
                model.eval()
                model.to("cpu")
                from torch.utils.data import DataLoader
                dl = DataLoader(ds, batch_size=ctx.cfg.eval_bs, shuffle=False, num_workers=0)
                import torch, math
                tot, correct, loss_sum, steps = 0, 0, 0.0, 0
                crit = torch.nn.CrossEntropyLoss()
                with torch.no_grad():
                    for x, y in dl:
                        logits = model(x.float())
                        loss = crit(logits, y)
                        loss_sum += float(loss.item())
                        pred = logits.argmax(dim=1)
                        correct += int((pred == y).sum().item())
                        tot += int(y.size(0))
                        steps += 1
                        if steps >= max_batches: break
                if tot == 0: return None, None, 0
                return (loss_sum/steps), (100.0*correct/tot), tot

            ds_val  = get_eval_dataset(ctx, "val")
            ds_test = get_eval_dataset(ctx, "test")
            calib_ds = ds_val or ds_test

            if os.getenv("FL_SANITY_LOCAL","0") == "1":
                l_loss, l_acc, ln = _quick_eval(local_state, ds_test or ds_val, max_batches=50)
                if ln:
                    logger.info(f"[AGG] sanity(local): acc={l_acc:.2f}% loss={l_loss:.4f} on ~{ln} samples")

            # 재보정 전 간이 정확도(작은 배치)로 BN 문제 가시화
            if os.getenv("FL_AGG_SANITY", "1") == "1":
                pre_loss, pre_acc, n_pre = _quick_eval(ctx.GLOBAL_MODEL_STATE, ds_test or ds_val, max_batches=50)
                if n_pre:
                    logger.info(f"[AGG] sanity(pre-BN): acc={pre_acc:.2f}% loss={pre_loss:.4f} on ~{n_pre} samples")

            # BN 재보정(강하게 권장)
            if calib_ds is not None:
                ctx.GLOBAL_MODEL_STATE = bn_recalibrate(
                    ctx.GLOBAL_MODEL_STATE, calib_ds,
                    batches=int(os.getenv("FL_BN_CALIB_BATCHES","600")),
                    bs=ctx.cfg.eval_bs, num_classes=ctx.cfg.num_classes
                )
                # 재보정 값 저장
                try:
                    torch.save(ctx.GLOBAL_MODEL_STATE, out_path)
                    torch.save(ctx.GLOBAL_MODEL_STATE, link_path)
                except Exception:
                    pass

            # 재보정 후 간이 정확도
            if os.getenv("FL_AGG_SANITY", "1") == "1":
                post_loss, post_acc, n_post = _quick_eval(ctx.GLOBAL_MODEL_STATE, ds_test or ds_val, max_batches=50)
                if n_post:
                    logger.info(f"[AGG] sanity(post-BN): acc={post_acc:.2f}% loss={post_loss:.4f} on ~{n_post} samples")

            # 정식 평가/로깅
            def _eval_and_log(split, ds):
                if ds is None: return
                key = (ctx.GLOBAL_MODEL_VERSION, split)
                with ctx.EVAL_DONE_LOCK:
                    if key in ctx.EVAL_DONE: return
                g_loss, g_acc, n, g_f1, g_madds, g_flops, g_lat = evaluate_state_dict(
                    ctx.GLOBAL_MODEL_STATE, ds, batch_size=ctx.cfg.eval_bs, device="cpu",
                    num_classes=ctx.cfg.num_classes
                )
                log_global_metrics(ctx.GLOBAL_MODEL_VERSION, split, g_loss, g_acc, n, str(out_path),
                                   f1_macro=g_f1, madds_M=g_madds, flops_M=g_flops, latency_ms=g_lat)
                with ctx.EVAL_DONE_LOCK:
                    ctx.EVAL_DONE.add((ctx.GLOBAL_MODEL_VERSION, split))

            _eval_and_log("val", ds_val)
            _eval_and_log("test", ds_test)

    except Exception as e:
        logger.warning(f"[AGG] global eval failed: {e}")

    return str(out_path)


async def upload_and_aggregate_async(ctx: AppState, sat_id: int, ckpt_path: str) -> str:
    import asyncio
    return await asyncio.to_thread(upload_and_aggregate, ctx, sat_id, ckpt_path)