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

def alpha_for(ctx: AppState, sat_id: int, s: int) -> float:
    if s >= max(ctx.cfg.fresh_cutoff, 0):
        return 0.0

    decay = staleness_factor(s, ctx.cfg.staleness_tau, ctx.cfg.staleness_mode)

    def _client_num_samples(_sid: int) -> int:
        try:
            ds = get_training_dataset(_sid)
            return len(ds) if hasattr(ds, "__len__") else 1
        except Exception:
            return 1

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

def upload_and_aggregate(ctx: AppState, sat_id: int, ckpt_path: str, *, n_samples: int | None = None) -> str:
    from collections import deque
    if not ckpt_path or not Path(ckpt_path).exists():
        raise FileNotFoundError(f"ckpt not found: {ckpt_path}")

    local_state = torch.load(ckpt_path, map_location="cpu")
    if any(isinstance(k, str) and k.startswith("module.") for k in local_state.keys()):
        local_state = {k.replace("module.", "", 1): v for k, v in local_state.items()}
    
    with ctx.GLOBAL_MODEL_LOCK:
        gver_before = ctx.GLOBAL_MODEL_VERSION
        base_ver = parse_fromg(ckpt_path) or gver_before
        staleness = max(0, gver_before - base_ver)
        if ctx.cfg.s_max_drop > 0 and staleness > ctx.cfg.s_max_drop:
            logger.info(f"[AGG] drop stale update: s={staleness} (> {ctx.cfg.s_max_drop}) from {ckpt_path}")
            ts_d = __import__("datetime").datetime.now().strftime("%Y%m%d_%H%M%S")
            out_d = CKPT_DIR / f"global_v{ctx.GLOBAL_MODEL_VERSION}_{ts_d}.ckpt"
            try: torch.save(ctx.GLOBAL_MODEL_STATE, out_d)
            except Exception: pass
            return str(out_d)

        # ----- α_eff 계산 (staleness/샘플 수 반영 + 클램프)
        ALPHA_BASE = float(os.getenv("FL_AGG_ALPHA", "0.05"))
        ALPHA_MIN  = float(os.getenv("FL_AGG_ALPHA_MIN", "0.01"))
        ALPHA_MAX  = float(os.getenv("FL_AGG_ALPHA_MAX", "0.08"))
        TAU        = float(os.getenv("FL_STALENESS_TAU", "1000"))
        W_MIN      = float(os.getenv("FL_STALENESS_W_MIN", "0.0"))
        MEAN_SAMP  = float(os.getenv("FL_MEAN_CLIENT_SAMPLES", "2000"))

        w_stale = max(W_MIN, math.exp(-staleness / max(TAU, 1e-9)))
        alpha_eff = ALPHA_BASE * w_stale
        if n_samples and n_samples > 0:
            alpha_eff *= min(1.0, n_samples / max(MEAN_SAMP, 1.0))
        alpha_eff = max(ALPHA_MIN, min(ALPHA_MAX, alpha_eff))

        # ----- 키 매칭/미스매치 분석
        g_state = ctx.GLOBAL_MODEL_STATE
        g_keys = set(g_state.keys())
        l_keys = set(local_state.keys())
        matched = [k for k in l_keys if (k in g_keys and hasattr(g_state[k], "shape") and hasattr(local_state[k], "shape")
                                         and g_state[k].shape == local_state[k].shape)]
        miss_g = sorted(list(g_keys - set(matched)))
        miss_l = sorted(list(l_keys - set(matched)))

        # 델타 노름 (집계 전 진단용)
        g_norm = _calc_norm(g_state, matched)
        d_norm = _calc_norm({k: (local_state[k].float() - g_state[k].float()) for k in matched}, matched)

        # ----- 실제 집계(매칭된 키만 가중합)
        if alpha_eff > 0.0 and matched:
            for k in matched:
                g = g_state[k].float()
                l = local_state[k].float()
                g_state[k] = ((1.0 - alpha_eff) * g + alpha_eff * l).to(g.dtype)

            ctx.GLOBAL_MODEL_STATE = g_state
            ctx.GLOBAL_MODEL_VERSION += 1

        # ----- 저장(타임스탬프 + 심볼릭 링크 역할 파일)
        ts = __import__("datetime").datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path  = CKPT_DIR / f"global_v{ctx.GLOBAL_MODEL_VERSION}_{ts}.ckpt"
        link_path = CKPT_DIR / f"global_v{ctx.GLOBAL_MODEL_VERSION}.ckpt"
        torch.save(ctx.GLOBAL_MODEL_STATE, out_path)
        try: torch.save(ctx.GLOBAL_MODEL_STATE, link_path)
        except Exception: pass
        save_last_global_ptr(out_path, ctx.GLOBAL_MODEL_VERSION)

        # ----- 집계 진단 로그 적재(최근 1000개 보관)
        if not hasattr(ctx, "AGG_LOG"):
            ctx.AGG_LOG = deque(maxlen=1000)
        ctx.AGG_LOG.append({
            "ts": ts,
            "sat_id": int(sat_id),
            "from_gver": int(base_ver),
            "gver_before": int(gver_before),
            "gver_after": int(ctx.GLOBAL_MODEL_VERSION),
            "alpha_eff": float(alpha_eff),
            "staleness": int(staleness),
            "w_stale": float(w_stale),
            "matched": int(len(matched)),
            "miss_g": int(len(miss_g)),
            "miss_l": int(len(miss_l)),
            "rel_delta": float(d_norm / (g_norm + 1e-12)),
            "ckpt": str(out_path),
        })


        logger.info(
            f"AGG sat={sat_id} g:{gver_before}->{ctx.GLOBAL_MODEL_VERSION} "
            f"from_g={base_ver} α_eff={alpha_eff:.4f} stl={staleness} "
            f"match={len(matched)} miss(g,l)=({len(miss_g)},{len(miss_l)}) "
            f"Δ‖W‖/‖W‖={ctx.AGG_LOG[-1]['rel_delta']:.3e} -> {out_path}"
        )

    # ----- 평가/로깅 (기존 로직 유지)
    try:
        if ctx.GLOBAL_MODEL_VERSION % max(1, ctx.cfg.eval_every_n) == 0:
            from .training import get_eval_dataset
            ds_val  = get_eval_dataset(ctx, "val")
            ds_test = get_eval_dataset(ctx, "test")
            calib_ds = ds_val or ds_test
            if calib_ds is not None:
                ctx.GLOBAL_MODEL_STATE = bn_recalibrate(
                    ctx.GLOBAL_MODEL_STATE, calib_ds,
                    batches=int(os.getenv("FL_BN_CALIB_BATCHES","200")),
                    bs=ctx.cfg.eval_bs, num_classes=ctx.cfg.num_classes
                )
                try:
                    torch.save(ctx.GLOBAL_MODEL_STATE, out_path)
                    torch.save(ctx.GLOBAL_MODEL_STATE, link_path)
                except Exception:
                    pass

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