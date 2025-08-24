# simserver/core/config.py
import os
from dataclasses import dataclass
from .utils import get_env_int, get_env_float
from .gpu import compute_num_gpus

@dataclass(frozen=True)
class Settings:
    # sim
    default_dl_workers: int
    dataloader_workers: int
    sim_paused: bool
    threshold_deg: float
    start_time_iso: str
    sim_delta_sec: float
    real_interval_sec: float

    # FL
    sessions_per_gpu: int
    max_train_workers: int
    eval_every_n: int
    eval_bs: int
    num_classes: int
    # staleness/aggregation
    agg_alpha: float
    momentum: float
    staleness_tau: float
    staleness_mode: str
    w_min: float
    s_max_drop: int
    alpha_max: float
    fresh_cutoff: int
    bn_scale: float

    # gpu
    num_gpus: int

def _is_wsl() -> bool:
    from pathlib import Path
    try:
        return "microsoft" in Path("/proc/version").read_text().lower()
    except Exception:
        return False

def load_settings() -> Settings:
    default_dl_workers = 0 if _is_wsl() or os.getenv("UVICORN_RELOAD") else 2
    dlw = get_env_int("FL_DATALOADER_WORKERS", default_dl_workers)

    env_num_gpus = os.getenv("NUM_GPUS")
    env_num_gpus = int(env_num_gpus) if env_num_gpus not in (None, "",) else None
    resolved_gpus = compute_num_gpus(env_override=env_num_gpus)

    sessions_per_gpu = get_env_int("SESSIONS_PER_GPU", 10)
    max_train_workers = get_env_int("FL_MAX_WORKERS", max(1, (resolved_gpus or 1) * max(1, sessions_per_gpu)))

    return Settings(
        default_dl_workers=default_dl_workers,
        dataloader_workers=dlw,
        sim_paused = False,
        threshold_deg=float(os.getenv("FL_ELEV_THRESHOLD_DEG", "40")),
        start_time_iso=os.getenv("FL_START_TIME_ISO", "2025-03-30T00:00:00"),
        sim_delta_sec=float(os.getenv("FL_SIM_DELTA_SEC", "10")),
        real_interval_sec=float(os.getenv("FL_REAL_INTERVAL_SEC", "0.01")),
        sessions_per_gpu=sessions_per_gpu,
        max_train_workers=max_train_workers,
        eval_every_n=get_env_int("FL_EVAL_EVERY_N", 1),
        eval_bs=get_env_int("FL_EVAL_BS", 1024),
        num_classes=get_env_int("FL_NUM_CLASSES", 10),
        agg_alpha=get_env_float("FL_AGG_ALPHA", 0.05),
        momentum=get_env_float("FL_SERVER_MOM", 0.9),
        staleness_tau=get_env_float("FL_STALENESS_TAU", 1000),
        staleness_mode=os.getenv("FL_STALENESS_MODE", "exp"),
        w_min=get_env_float("FL_STALENESS_W_MIN", 0.0),
        s_max_drop=get_env_int("FL_STALENESS_MAX_DROP", 64),
        alpha_max=get_env_float("FL_AGG_ALPHA_MAX", 0.05),
        fresh_cutoff=get_env_int("FL_STALENESS_FRESH_CUTOFF", 40),
        bn_scale=get_env_float("FL_AGG_BN_SCALE", 0.1),
        num_gpus=resolved_gpus,
    )
