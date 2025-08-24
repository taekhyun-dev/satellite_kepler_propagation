# sat_comm_server.py
from __future__ import annotations

import os, sys, atexit
import json
import asyncio
import threading
import logging
import copy
import time
import importlib
import math, re
import traceback

from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, Future
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from collections import Counter

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse
from skyfield.api import load, EarthSatellite, Topos
from pathlib import Path

# --- data registry (로컬 CIFAR 분배 모듈) ---
from data import (
    CIFAR_ROOT, ASSIGNMENTS_DIR, SAMPLES_PER_CLIENT, DIRICHLET_ALPHA, RNG_SEED, WITH_REPLACEMENT,
    DATA_REGISTRY, get_training_dataset,
)

# -------------------- Optional: torch --------------------
try:
    import torch, torch.nn as nn
    _has_torch = True
except Exception:
    _has_torch = False

if _has_torch:
    import torch.multiprocessing as mp
    try:
        if mp.get_start_method(allow_none=True) != "spawn":
            mp.set_start_method("spawn", force=True)
        torch.multiprocessing.set_sharing_strategy("file_system")
    except RuntimeError:
        pass

# -------------------- Logger --------------------
logger = logging.getLogger("simserver")
logger.setLevel(logging.INFO)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(h)

def _log(msg: str):
    logger.info(f"[FL] {msg}")

def _exc_repr(e: Exception) -> str:
    try:
        return f"{type(e).__name__}: {e}"
    except Exception:
        return f"{type(e).__name__}"

def _is_zip_file(path: Path) -> bool:
    try:
        import zipfile;  return zipfile.is_zipfile(path)
    except Exception:     return False

# -------------------- 환경/경로 --------------------
def _is_wsl() -> bool:
    try:
        return "microsoft" in Path("/proc/version").read_text().lower()
    except Exception:
        return False

DEFAULT_DL_WORKERS = 0 if _is_wsl() or os.getenv("UVICORN_RELOAD") else 2
DL_WORKERS = int(os.getenv("FL_DATALOADER_WORKERS", str(DEFAULT_DL_WORKERS)))

BASE_DIR = Path(__file__).resolve().parent
CKPT_DIR = BASE_DIR / "ckpt"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

LAST_GLOBAL_PTR = CKPT_DIR / "last_global.json"

# --- Metrics paths ---
METRICS_DIR = BASE_DIR / "metrics"
GLOBAL_METRICS_DIR = METRICS_DIR / "global"
GLOBAL_METRICS_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_METRICS_DIR = METRICS_DIR / "local"
LOCAL_METRICS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

GLOBAL_METRICS_CSV = GLOBAL_METRICS_DIR / "global_metrics.csv"
GLOBAL_METRICS_XLSX = GLOBAL_METRICS_DIR / "global_metrics.xlsx"
GLOBAL_METRICS_HEADER = [
    "timestamp", "version", "split", "num_samples",
    "loss", "acc", "f1_macro", "madds_M", "flops_M", "latency_ms",
    "ckpt_path"
]

try:
    import pandas as _pd  # optional
    _has_pandas = True
except Exception:
    _has_pandas = False

METRICS_LOCK = threading.Lock()

# ---- 글로벌 모델 집계 상태 ----
GLOBAL_MODEL_LOCK = threading.Lock()
GLOBAL_MODEL_STATE = None
GLOBAL_MODEL_VERSION = -1

# 튜닝 환경변수
AGG_ALPHA     = float(os.getenv("FL_AGG_ALPHA", "0.25"))      # 기본 집계 비율
STALENESS_TAU = float(os.getenv("FL_STALENESS_TAU", "14"))   # 감쇠 속도
STALENESS_MODE= os.getenv("FL_STALENESS_MODE", "exp")        # "exp" or "poly"
W_MIN         = float(os.getenv("FL_STALENESS_W_MIN", "0.0"))# 바닥 가중치(신선 update에만)
S_MAX_DROP    = int(os.getenv("FL_STALENESS_MAX_DROP", "64"))# s 너무 크면 드랍
ALPHA_MAX     = float(os.getenv("FL_AGG_ALPHA_MAX", "0.5"))
FRESH_CUTOFF  = int(os.getenv("FL_STALENESS_FRESH_CUTOFF", "40"))

# 기타 평가 주기
EVAL_EVERY_N = int(os.getenv("FL_EVAL_EVERY_N", "2"))
EVAL_BS      = int(os.getenv("FL_EVAL_BS", "1024"))

# --- Eval caching / dedup ---
EVAL_DS_CACHE = {"val": None, "test": None}
EVAL_FALLBACK_LOGGED = set()
EVAL_DONE = set()
EVAL_DONE_LOCK = threading.Lock()

_fromg_re = re.compile(r"fromg(\d+)")


# ==================== 시뮬레이션 상태 변수 ====================
satellites: Dict[int, EarthSatellite] = {}
sat_comm_status: Dict[int, bool] = {}
current_sat_positions: Dict[int, Dict[str, float]] = {}

observer_locations = {
    "Berlin":  Topos(latitude_degrees=52.52,  longitude_degrees=13.41,  elevation_m=34),
    "Houston": Topos(latitude_degrees=29.76,  longitude_degrees=-95.37, elevation_m=30),
    "Tokyo":   Topos(latitude_degrees=35.68,  longitude_degrees=139.69, elevation_m=40),
    "Nairobi": Topos(latitude_degrees=-1.29,  longitude_degrees=36.82,  elevation_m=1700),
    "Sydney":  Topos(latitude_degrees=-33.87, longitude_degrees=151.21, elevation_m=58),
}
raw_iot_clusters = {
    "Abisko":         {"latitude": 68.35,  "longitude": 18.79,  "elevation_m": 420},
    "Boreal":         {"latitude": 55.50,  "longitude": 105.00, "elevation_m": 450},
    "Taiga":          {"latitude": 58.00,  "longitude": 99.00,  "elevation_m": 300},
    "Patagonia":      {"latitude": 51.00,  "longitude": 73.00,  "elevation_m": 500},
    "Amazon_Forest":  {"latitude": -3.47,  "longitude": -62.37, "elevation_m": 100},
    "Great_Barrier":  {"latitude": -18.29, "longitude": 147.77, "elevation_m": 0},
    "Mediterranean":  {"latitude": 37.98,  "longitude": 23.73,  "elevation_m": 170},
    "California":     {"latitude": 36.78,  "longitude": -119.42,"elevation_m": 150},
}
iot_clusters = {
    name: Topos(latitude_degrees=cfg["latitude"], longitude_degrees=cfg["longitude"], elevation_m=cfg["elevation_m"])
    for name, cfg in raw_iot_clusters.items()
}

current_observer_name = "Berlin"
observer = observer_locations[current_observer_name]

ts = load.timescale()
sim_time = datetime(2025, 3, 30, 0, 0, 0)
threshold_deg = 40
sim_paused = False
auto_resume_delay_sec = 0
sim_delta_sec = 10.0
real_interval_sec = 0.01

def to_ts(dt: datetime):
    return ts.utc(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second)

def get_current_time_utc():
    return to_ts(sim_time)

def elevation_deg(sat: EarthSatellite, topox: Topos, t_ts):
    alt, _, _ = (sat - topox).at(t_ts).altaz()
    return alt.degrees


# ==================== 연합학습(FL) 런타임 상태 ====================
# GPU 개수 (환경 그대로 존중: CUDA_VISIBLE_DEVICES 를 사용자가 설정 가능)
NUM_GPUS = 0
if _has_torch and torch.cuda.is_available():
    NUM_GPUS = torch.cuda.device_count()

SESSIONS_PER_GPU = int(os.getenv("SESSIONS_PER_GPU", "10"))
MAX_TRAIN_WORKERS = int(os.getenv("FL_MAX_WORKERS", str(max(1, (NUM_GPUS or 1) * max(1, SESSIONS_PER_GPU)))))

training_executor = ThreadPoolExecutor(max_workers=MAX_TRAIN_WORKERS)
uploader_executor = ThreadPoolExecutor(max_workers=4)

train_queue: "asyncio.Queue[int]" = asyncio.Queue(
    maxsize=int(os.getenv("FL_QUEUE_MAX", "1000"))
)

@dataclass
class TrainState:
    running: bool = False
    future: Optional[Future] = None
    gpu_id: Optional[int] = None
    stop_event: threading.Event = field(default_factory=threading.Event)
    last_ckpt_path: Optional[str] = None
    round_idx: int = 0
    in_queue: bool = False

train_states: Dict[int, TrainState] = {}

# --- GPU 선택 이후에 호출될 유틸들 ---------------------
def _count_visible_gpus_from_env() -> int | None:
    cvd = os.getenv("CUDA_VISIBLE_DEVICES")
    if cvd is None:
        return None                      # 환경변수 미지정 → 모름
    cvd = cvd.strip()
    if cvd == "":
        return 0                         # 강제로 CPU
    return len([x for x in cvd.split(",") if x.strip() != ""])

def compute_num_gpus() -> int:
    # 0) 사용자가 NUM_GPUS를 명시하면 그 값을 우선하되, 보이는 GPU 개수로 cap
    env_num = os.getenv("NUM_GPUS")
    if env_num is not None:
        try:
            wanted = max(0, int(env_num))
        except ValueError:
            wanted = 0
        vis = _count_visible_gpus_from_env()
        return min(wanted, vis) if vis is not None else wanted

    # 1) CUDA_VISIBLE_DEVICES가 설정되어 있으면 그 개수 사용
    vis = _count_visible_gpus_from_env()
    if vis is not None:
        # 토치가 CUDA를 못 쓰는 상황(드라이버/런타임 불일치)이면 안전하게 0
        try:
            import torch
            if not torch.cuda.is_available():
                return 0
        except Exception:
            pass
        return vis

    # 2) 마지막으로 물리 GPU 개수 (환경변수 미설정 시)
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.device_count()
    except Exception:
        pass
    return 0

# 2) 이제 보이는 GPU 기준으로 NUM_GPUS 계산
NUM_GPUS = compute_num_gpus()
# ---------------------------------------------------------

def _save_last_global_ptr(path: Path, version: int):
    try:
        tmp = LAST_GLOBAL_PTR.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump({"path": str(path), "version": int(version)}, f)
        tmp.replace(LAST_GLOBAL_PTR)
    except Exception as e:
        logger.warning(f"[AGG] write last_global.json failed: {e}")

def _load_last_global_ptr() -> tuple[int, Optional[Path]]:
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

def _find_latest_global_ckpt():
    paths = list(CKPT_DIR.glob("global_v*.ckpt"))
    best_ver, best_ts, best_path = -1, "", None
    for p in paths:
        m = re.match(r"global_v(\d+)(?:_(\d{8}_\d{6}))?\.ckpt$", p.name)
        if not m:
            continue
        v = int(m.group(1))
        ts = m.group(2) or ""
        if (v > best_ver) or (v == best_ver and ts > best_ts):
            best_ver, best_ts, best_path = v, ts, p
    return best_ver, best_path

def _local_ptr_path(sat_id: int) -> Path:
    return CKPT_DIR / f"sat{sat_id}_last.json"

def _save_last_local_ptr(sat_id: int, path: str, *, from_gver: Optional[int] = None, round_idx: Optional[int] = None, epoch: Optional[int] = None):
    try:
        p = _local_ptr_path(sat_id)
        tmp = p.with_suffix(".json.tmp")
        meta = {
            "path": str(path),
            "from_gver": int(from_gver) if from_gver is not None else (_parse_fromg(os.path.basename(path)) or -1),
            "round_idx": int(round_idx) if round_idx is not None else None,
            "epoch": int(epoch) if epoch is not None else None,
            "updated_at": datetime.now().isoformat(),
        }
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(meta, f)
        tmp.replace(p)
    except Exception as e:
        logger.warning(f"[LOCAL] write last ptr failed (sat{sat_id}): {e}")

def _load_last_local_ptr(sat_id: int) -> Optional[Dict[str, Any]]:
    try:
        p = _local_ptr_path(sat_id)
        if not p.exists():
            return None
        with p.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        path = meta.get("path")
        if path and os.path.exists(path):
            return meta
    except Exception as e:
        logger.warning(f"[LOCAL] read last ptr failed (sat{sat_id}): {e}")
    return None

def _enqueue_training(sat_id: int):
    st = train_states[sat_id]
    if st.running or st.in_queue:
        return
    try:
        train_queue.put_nowait(sat_id)
        st.in_queue = True
    except asyncio.QueueFull:
        pass

def _model_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")

def _move_optimizer_state(optimizer: torch.optim.Optimizer, device: torch.device):
    for state in optimizer.state.values():
        for k, v in list(state.items()):
            if torch.is_tensor(v):
                state[k] = v.to(device, non_blocking=True)

def _ensure_all_on_model_device(model, optimizer, criterion=None):
    dev = _model_device(model)
    model.to(dev)
    if criterion is not None:
        # CrossEntropyLoss(weight=...) 같은 경우 weight 텐서도 같이 이동해야 함
        if hasattr(criterion, 'weight') and torch.is_tensor(criterion.weight):
            criterion.weight = criterion.weight.to(dev, non_blocking=True)
        try:
            criterion.to(dev)
        except Exception:
            pass
    if optimizer is not None:
        _move_optimizer_state(optimizer, dev)
    return dev

def _build_worker_gpu_list():
    if NUM_GPUS > 0:
        return [gid for gid in range(NUM_GPUS) for _ in range(SESSIONS_PER_GPU)]
    return [None] * max(1, SESSIONS_PER_GPU)

async def _train_worker(gpu_id: Optional[int]):
    loop = asyncio.get_running_loop()
    while True:
        ckpt = None  # <- 미리 초기화

        sat_id = await train_queue.get()
        st = train_states[sat_id]
        st.in_queue = False
        if st.running:
            train_queue.task_done()
            continue

        st.stop_event.clear()
        st.gpu_id = gpu_id
        st.running = True

        def _job():
            return do_local_training(sat_id=sat_id, stop_event=st.stop_event, gpu_id=gpu_id)

        fut = loop.run_in_executor(training_executor, _job)
        st.future = fut

        try:
            ckpt = await fut
        except Exception as e:
            tb = traceback.format_exc()
            _log(f"SAT{sat_id}: training ERROR: {e}\n{tb}")
            ckpt = None
        finally:
            st.last_ckpt_path = ckpt
            st.running = False
            _log(f"SAT{sat_id}: training DONE (worker gpu={gpu_id}), ckpt={ckpt}")
            train_queue.task_done()

def _upload_and_aggregate_async(sat_id: int, ckpt_path: Optional[str]):
    if not ckpt_path:
        _log(f"SAT{sat_id}: no checkpoint to upload")
        return

    def _job():
        _log(f"SAT{sat_id}: uploading {ckpt_path}")
        try:
            if "upload_model_to_server" in globals():
                upload_model_to_server(sat_id=sat_id, ckpt_path=ckpt_path)
                _log(f"SAT{sat_id}: uploaded")
            if "upload_and_aggregate" in globals():
                new_global = upload_and_aggregate(sat_id, ckpt_path)
                _log(f"SAT{sat_id}: aggregated -> {new_global}")
        except Exception as e:
            _log(f"SAT{sat_id}: upload/aggregate ERROR: {e}")

    uploader_executor.submit(_job)

def _on_become_visible(sat_id: int):
    st = train_states[sat_id]
    if st.running and st.future:
        st.stop_event.set()
        def _cb(_fut):
            try:
                _fut.result()
            except Exception:
                pass
            _upload_and_aggregate_async(sat_id, train_states[sat_id].last_ckpt_path)
        try:
            st.future.add_done_callback(_cb)
        except Exception:
            try:
                asyncio.ensure_future(st.future).add_done_callback(_cb)
            except Exception:
                pass
    else:
        _upload_and_aggregate_async(sat_id, st.last_ckpt_path)

# ---------- Metrics helpers ----------
def _append_csv_row(path: Path, header: List[str], row: List[Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with METRICS_LOCK:
        with path.open("a", encoding="utf-8") as f:
            if new_file:
                f.write(",".join(header) + "\n")
            def esc(x):
                s = "" if x is None else str(x)
                if ("," in s) or ("\"" in s):
                    s = "\"" + s.replace("\"", "\"\"") + "\""
                return s
            f.write(",".join(esc(x) for x in row) + "\n")

def _append_excel_row(xlsx_path: Path, header: List[str], row: List[Any], sheet: str = "metrics"):
    if not _has_pandas or os.getenv("FL_DISABLE_XLSX", "0") in ("1","true","True"):
        return
    xlsx_path.parent.mkdir(parents=True, exist_ok=True)
    import pandas as pd
    try:
        df = None
        if xlsx_path.exists():
            if not _is_zip_file(xlsx_path):
                try: xlsx_path.unlink()
                except Exception: pass
            else:
                try:
                    df = pd.read_excel(xlsx_path, sheet_name=sheet, engine="openpyxl")
                except Exception:
                    df = pd.DataFrame(columns=header)
        if df is None:
            df = pd.DataFrame(columns=header)
        s = pd.Series(row, index=header)
        df = pd.concat([df, s.to_frame().T], ignore_index=True)
        with pd.ExcelWriter(xlsx_path, engine="openpyxl", mode="w") as w:
            df.to_excel(w, sheet_name=sheet, index=False)
    except Exception as e:
        logger.warning(f"[METRICS] Excel write failed: {_exc_repr(e)}")

def _log_local_metrics(sat_id: int, round_idx: int, epoch: int, loss: float, acc: float, n: int, ckpt_path: str):
    ts = datetime.now().isoformat()
    header = ["timestamp", "sat_id", "round", "epoch", "num_samples", "loss", "acc", "ckpt_path"]
    row = [ts, sat_id, round_idx, epoch, n, f"{loss:.6f}", f"{acc:.2f}", ckpt_path]
    csv_path = LOCAL_METRICS_DIR / f"sat{sat_id}.csv"
    _append_csv_row(csv_path, header, row)
    _append_excel_row(csv_path.with_suffix(".xlsx"), header, row, sheet="local")

def _log_global_metrics(
    version: int,
    split: str,
    loss: float,
    acc: float,
    n: int,
    ckpt_path: str,
    *,
    f1_macro: float = None,
    madds_M: float = None,
    flops_M: float = None,
    latency_ms: float = None,
):
    ts = datetime.now().isoformat()
    row = [
        ts,
        version,
        split,
        n,
        f"{loss:.6f}",
        f"{acc:.2f}",
        (f"{f1_macro:.2f}" if f1_macro is not None else ""),
        (f"{madds_M:.2f}"  if madds_M is not None  else ""),
        (f"{flops_M:.2f}"  if flops_M is not None  else ""),
        (f"{latency_ms:.3f}" if latency_ms is not None else ""),
        ckpt_path,
    ]
    _append_csv_row(GLOBAL_METRICS_CSV, GLOBAL_METRICS_HEADER, row)
    try:
        _append_excel_row(GLOBAL_METRICS_XLSX, GLOBAL_METRICS_HEADER, row, sheet="global")
    finally:
        _log(f"[METRICS] global v{version} {split} saved -> {GLOBAL_METRICS_CSV}")

def _client_num_samples(sat_id: int) -> int:
    try:
        ds = get_training_dataset(sat_id)
        return len(ds) if hasattr(ds, "__len__") else 1
    except Exception:
        return 1
    
def _staleness_factor(s: int, tau: float, mode: str) -> float:
    if tau <= 0: return 1.0
    if mode == "poly":
        return (1.0 + s) ** (-tau)
    # default exp
    return math.exp(-float(s) / tau)

def _init_global_model():
    """서버 기동 시 글로벌 가중치 로드/초기화."""
    global GLOBAL_MODEL_STATE, GLOBAL_MODEL_VERSION
    import torch

    with GLOBAL_MODEL_LOCK:
        # 1) 포인터 파일 우선
        ver, path = _load_last_global_ptr()
        # 2) 없으면 디렉토리 스캔
        if path is None:
            ver, path = _find_latest_global_ckpt()

        if path and path.exists() and ver >= 0:
            try:
                GLOBAL_MODEL_STATE = torch.load(path, map_location="cpu")
                GLOBAL_MODEL_VERSION = ver
                print(f"[AGG] Loaded global model v{GLOBAL_MODEL_VERSION} from {path}")
                return
            except Exception as e:
                print(f"[AGG] Failed to load existing global ({path}): {e}. Re-initializing...")

        # 새로 생성해서 v0 저장
        model = _new_model_skeleton()
        GLOBAL_MODEL_STATE = model.state_dict()
        GLOBAL_MODEL_VERSION = 0

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        init_path = CKPT_DIR / f"global_v0_{ts}.ckpt"
        link_path = CKPT_DIR / "global_v0.ckpt"
        try:
            torch.save(GLOBAL_MODEL_STATE, init_path)
            torch.save(GLOBAL_MODEL_STATE, link_path)
        except Exception as e:
            print(f"[AGG] Initial save warning: {e}")
        _save_last_global_ptr(init_path, 0)
        print(f"[AGG] Initialized new global model at {init_path}")


def _new_model_skeleton():
    """학습과 동일한 아키텍처로 빈 모델 생성."""
    from torchvision.models import mobilenet_v3_small
    num_classes = int(os.getenv("FL_NUM_CLASSES", "10"))
    model = mobilenet_v3_small(num_classes=num_classes)
    return model

# --- 크기/신선도 가중 α 계산
def _alpha_for(sat_id: int, s: int) -> float:
    """
    s: staleness = (current_global_version - base_global_version_from_client)
    α_eff = α_base * decay(s) * size_scale, 단 지나치게 오래되면 0.
    """
    AGG_ALPHA     = _get_env_float("FL_AGG_ALPHA", 0.2)
    STALENESS_TAU = _get_env_float("FL_STALENESS_TAU", 14.0)
    STALENESS_MODE= os.getenv("FL_STALENESS_MODE", "exp")
    W_MIN         = _get_env_float("FL_STALENESS_W_MIN", 0.0)
    ALPHA_MAX     = _get_env_float("FL_AGG_ALPHA_MAX", 0.5)
    FRESH_CUTOFF  = _get_env_int("FL_STALENESS_FRESH_CUTOFF", 40)

    # 신선도 컷오프
    if s >= max(FRESH_CUTOFF, 0):
        return 0.0

    decay = _staleness_factor(s, STALENESS_TAU, STALENESS_MODE)

    # 데이터 크기 보정
    try:
        # satellites 전역이 있다면 평균 샘플 수를 사용
        mean_n = sum(_client_num_samples(sid) for sid in satellites) / max(1, len(satellites))
    except Exception:
        mean_n = _client_num_samples(sat_id)

    scale = _client_num_samples(sat_id) / max(1e-9, mean_n)

    alpha_eff = float(AGG_ALPHA * decay * scale)
    if W_MIN > 0.0:
        alpha_eff = max(W_MIN, alpha_eff)
    alpha_eff = min(alpha_eff, ALPHA_MAX)
    return float(alpha_eff)

# --- 파라미터 합성(글로벌 <- 로컬) : BN 러닝스탯/카운트 안전 처리
def aggregate_params(global_state: dict, local_state: dict, alpha: float) -> dict:
    """
    new = (1 - alpha) * global + alpha * local
    - 키가 없거나 shape가 다르면 글로벌 값 유지
    - BN running_*는 축소된 alpha로 blend (기본 0.1배), num_batches_tracked는 글로벌 유지
    """
    import torch

    out = {}
    alpha_bn_scale = _get_env_float("FL_AGG_BN_SCALE", 0.1)

    for k, g_t in global_state.items():
        l_t = local_state.get(k)
        if l_t is None or getattr(g_t, "shape", None) != getattr(l_t, "shape", None):
            out[k] = g_t.clone() if hasattr(g_t, "clone") else g_t
            continue

        # BN running stats
        if k.endswith("running_mean") or k.endswith("running_var"):
            if hasattr(g_t, "is_floating_point") and g_t.is_floating_point() and l_t.is_floating_point():
                g = g_t.detach().to("cpu", dtype=torch.float32)
                l = l_t.detach().to("cpu", dtype=torch.float32)
                a = float(alpha) * alpha_bn_scale
                out[k] = ((1.0 - a) * g + a * l).to(dtype=g_t.dtype)
            else:
                out[k] = g_t.clone()
            continue

        if k.endswith("num_batches_tracked"):
            out[k] = g_t.clone()
            continue

        # 일반 파라미터
        if hasattr(g_t, "is_floating_point") and g_t.is_floating_point() and l_t.is_floating_point():
            g = g_t.detach().to("cpu", dtype=torch.float32)
            l = l_t.detach().to("cpu", dtype=torch.float32)
            out[k] = ((1.0 - alpha) * g + alpha * l).to(dtype=g_t.dtype)
        else:
            out[k] = g_t.clone() if hasattr(g_t, "clone") else g_t

    return out

# --- 평가용 데이터셋 로딩 (레지스트리 → data 모듈 → CIFAR10 폴백)
def _get_eval_dataset(split: str):
    """
    split in {"val","test"}.
    우선순위:
      1) DATA_REGISTRY.get_validation/test_dataset()
      2) data.get_validation_dataset / get_test_dataset
      3) torchvision CIFAR-10(test) with resize+norm
    결과는 EVAL_DS_CACHE에 캐싱.
    """
    # 캐시
    if split in EVAL_DS_CACHE and EVAL_DS_CACHE[split] is not None:
        return EVAL_DS_CACHE[split]

    # 1) DATA_REGISTRY 메서드
    try:
        if "DATA_REGISTRY" in globals():
            meth_name = f"get_{'validation' if split=='val' else 'test'}_dataset"
            meth = getattr(DATA_REGISTRY, meth_name, None)
            if callable(meth):
                ds = meth()
                if ds is not None:
                    EVAL_DS_CACHE[split] = ds
                    return ds
    except Exception as e:
        logger.warning(f"[EVAL] DATA_REGISTRY method failed ({split}): {e}")

    # 2) data 모듈 함수
    try:
        data_mod = importlib.import_module("data")
        fn_name = "get_validation_dataset" if split == "val" else "get_test_dataset"
        fn = getattr(data_mod, fn_name, None)
        if callable(fn):
            ds = fn()
            if ds is not None:
                EVAL_DS_CACHE[split] = ds
                return ds
    except Exception as e:
        logger.warning(f"[EVAL] data.{split} fallback failed: {e}")

    # 3) torchvision CIFAR10(test)
    try:
        from torchvision.datasets import CIFAR10
        from torchvision import transforms

        img_size = _get_env_int("FL_IMG_SIZE", 32)
        mean = tuple(map(float, os.getenv("FL_NORM_MEAN", "0.4914,0.4822,0.4465").split(",")))
        std  = tuple(map(float, os.getenv("FL_NORM_STD",  "0.2470,0.2435,0.2616").split(",")))

        tfm = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        # CIFAR_ROOT가 없으면 ./data 로
        try:
            root_dir = str(CIFAR_ROOT)
        except NameError:
            root_dir = str(Path.cwd() / "data")

        ds = CIFAR10(root=root_dir, train=False, download=True, transform=tfm)
        if "fallback" not in EVAL_FALLBACK_LOGGED:
            _log(f"[EVAL] fallback CIFAR10(test) | size={img_size}, norm=mean{mean},std{std}")
            EVAL_FALLBACK_LOGGED.add("fallback")
        EVAL_DS_CACHE[split] = ds
        return ds
    except Exception as e:
        logger.warning(f"[EVAL] torchvision CIFAR10 fallback failed: {e}")
        return None

# 안전 가드: 외부 전역이 없을 때 기본값 준비
try:
    EVAL_DS_CACHE
except NameError:
    EVAL_DS_CACHE = {"val": None, "test": None}
try:
    EVAL_FALLBACK_LOGGED
except NameError:
    EVAL_FALLBACK_LOGGED = set()

# env 기반 하이퍼파라미터(이미 전역이 있다면 그것을 사용)
def _get_env_float(name, default):
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return float(default)

def _get_env_int(name, default):
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return int(default)

# --- if _staleness_factor가 없으면 정의 (있으면 이 블록은 무시됨)
try:
    _staleness_factor
except NameError:
    import math
    def _staleness_factor(s: int, tau: float, mode: str = "exp") -> float:
        if tau <= 0: return 1.0
        if mode == "poly":
            return (1.0 + s) ** (-tau)
        return math.exp(-float(s) / tau)

# --- 파일명에서 from_gver 추출
def _parse_fromg(ckpt_path: str) -> int | None:
    """
    예: 'sat3_fromg12_round1_ep0_20250818_145232.ckpt' -> 12
    """
    m = re.search(r"fromg(\d+)", os.path.basename(ckpt_path))
    return int(m.group(1)) if m else None

# --- 클라이언트 샘플수 (없으면 1)
try:
    _client_num_samples
except NameError:
    def _client_num_samples(_sat_id: int) -> int:
        return 1

# --- state_dict 평가기 (loss/acc/F1, latency, FLOPs)
def _evaluate_state_dict(state_dict: dict, dataset, batch_size: int = 512, device: str = "cpu"):
    """
    주어진 state_dict를 dataset 위에서 평가.
    반환: (avg_loss, acc%, total_samples, f1_macro%, madds_M, flops_M, latency_ms)
    """
    import torch, torch.nn as nn
    from torch.utils.data import DataLoader
    from torchvision.models import mobilenet_v3_small

    dev = torch.device(device) if not isinstance(device, torch.device) else device
    num_classes = _get_env_int("FL_NUM_CLASSES", 10)

    model = mobilenet_v3_small(num_classes=num_classes)
    model.load_state_dict(state_dict, strict=False)
    model.to(dev).eval()

    pin_mem = (isinstance(dev, torch.device) and dev.type == "cuda")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin_mem)
    criterion = nn.CrossEntropyLoss(reduction="sum")

    total_loss, total_correct, total = 0.0, 0, 0
    conf = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    with torch.no_grad():
        for x, y in loader:
            x = x.to(dev, non_blocking=pin_mem)
            y = y.to(dev, dtype=torch.long, non_blocking=pin_mem)

            logits = model(x)
            loss = criterion(logits, y)
            total_loss += float(loss.item())

            pred = logits.argmax(1)
            total_correct += int((pred == y).sum().item())
            total += y.size(0)

            # 클래스별 bincount로 혼동행렬 누적
            for c in range(num_classes):
                m = (y == c)
                if m.any():
                    conf[c] += torch.bincount(pred[m].detach().cpu(), minlength=num_classes).to(conf.dtype)

    avg_loss = total_loss / max(1, total)
    acc = 100.0 * total_correct / max(1, total)

    # F1-macro
    tp = conf.diag().to(torch.float32)
    fp = conf.sum(0).to(torch.float32) - tp
    fn = conf.sum(1).to(torch.float32) - tp
    precision = tp / (tp + fp + 1e-12)
    recall    = tp / (tp + fn + 1e-12)
    f1_macro  = float((2 * precision * recall / (precision + recall + 1e-12)).mean().item() * 100.0)

    # Latency (bs=1)
    try:
        sample_shape = dataset[0][0].shape  # (C,H,W)
    except Exception:
        sample_shape = (3, 32, 32)

    x1 = torch.randn(1, *sample_shape, device=dev)
    with torch.no_grad():
        # warmup
        for _ in range(5):
            _ = model(x1)
        if isinstance(dev, torch.device) and dev.type == "cuda":
            torch.cuda.synchronize(dev)
        t0 = time.perf_counter()
        for _ in range(20):
            _ = model(x1)
        if isinstance(dev, torch.device) and dev.type == "cuda":
            torch.cuda.synchronize(dev)
        latency_ms = (time.perf_counter() - t0) / 20.0 * 1000.0

    # FLOPs / MAdds (best-effort)
    madds_M, flops_M = None, None
    try:
        from thop import profile
        macs, _params = profile(model, inputs=(x1,), verbose=False)
        madds_M = macs / 1e6
        flops_M = (2 * macs) / 1e6
    except Exception:
        try:
            from fvcore.nn import FlopCountAnalysis
            model_cpu = model.to("cpu")
            x_cpu = x1.detach().to("cpu")
            flop_an = FlopCountAnalysis(model_cpu, x_cpu)
            flops_M = float(flop_an.total()) / 1e6
            madds_M = flops_M / 2.0
            model.to(dev)
        except Exception:
            pass

    return avg_loss, acc, total, f1_macro, madds_M, flops_M, float(latency_ms)

def _bn_recalibrate(state_dict: dict, dataset, batches: int = 20, bs: int = 256, device: Optional[str] = None) -> dict:
    """
    BatchNorm 통계(running_mean/var)만 다시 맞추기 위한 짧은 패스.
    - state_dict를 로드한 동일 아키텍처 모델을 train() 모드로 두고,
      몇 개 배치만 순전파하여 BN 통계 업데이트.
    - 가중치는 변경되지 않으며, 추론 시 정확도 하락을 막는 데 도움.
    반환: recalibrated state_dict (CPU 텐서)
    """
    if dataset is None:
        # 데이터셋 없으면 원본 반환
        return state_dict

    import torch
    from torch.utils.data import DataLoader

    # 모델 준비: 프로젝트에 _new_model_skeleton()이 있으면 우선 사용
    try:
        model = _new_model_skeleton()
    except Exception:
        from torchvision.models import mobilenet_v3_small
        num_classes = int(os.getenv("FL_NUM_CLASSES", "10"))
        model = mobilenet_v3_small(num_classes=num_classes)

    # state_dict 로드 (shape 불일치 허용)
    try:
        model.load_state_dict(state_dict, strict=False)
    except Exception:
        # 일부 키 타입/디바이스 이슈 대비
        fixed = {k: (v.detach().cpu() if hasattr(v, "detach") else v) for k, v in state_dict.items()}
        model.load_state_dict(fixed, strict=False)

    # 디바이스 선택
    if device is None:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device) if not isinstance(device, torch.device) else device

    model.to(dev)
    model.train()  # BN 통계 업데이트를 위해 train 모드 필요

    # 로더 구성
    pin = (isinstance(dev, torch.device) and dev.type == "cuda")
    loader = DataLoader(
        dataset,
        batch_size=int(bs),
        shuffle=True,
        num_workers=0,          # 안정성 우선
        pin_memory=pin,
        drop_last=False,
    )

    # 몇 배치만 통과시키면 충분
    seen = 0
    # 그래디언트/그래프 불필요 → inference_mode로 빠르게
    with torch.inference_mode():
        for x, *_ in loader:
            x = x.to(dev, non_blocking=pin)
            _ = model(x)
            seen += 1
            if seen >= int(batches):
                break

    # 평가 모드 전환 후 CPU state_dict 반환
    model.eval()
    recal = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return recal

def upload_and_aggregate(sat_id: int, ckpt_path: str) -> str:
    """
    위성에서 올라온 로컬 ckpt(=state_dict)를 글로벌에 합치고,
    새로운 글로벌 ckpt 경로를 반환.
    """
    import torch
    import datetime as _dt
    if not ckpt_path or not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"ckpt not found: {ckpt_path}")

    local_state = torch.load(ckpt_path, map_location="cpu")

    with GLOBAL_MODEL_LOCK:
        global GLOBAL_MODEL_STATE, GLOBAL_MODEL_VERSION

        # [3] staleness-aware α/size 보정
        base_ver = _parse_fromg(ckpt_path)
        s = max(0, GLOBAL_MODEL_VERSION - (base_ver or GLOBAL_MODEL_VERSION))
        if S_MAX_DROP > 0 and s > S_MAX_DROP:
            _log(f"[AGG] drop stale update: s={s} (> {S_MAX_DROP}) from {ckpt_path}")
            return str(CKPT_DIR / f"global_v{GLOBAL_MODEL_VERSION}.ckpt")
        
        # decay = _staleness_factor(s, STALENESS_TAU, STALENESS_MODE)
        alpha_eff = _alpha_for(sat_id, s)
        if S_MAX_DROP > 0 and s > S_MAX_DROP:
            _log(f"[AGG] drop stale update: s={s} (> {S_MAX_DROP}) from {ckpt_path}")
            return str(CKPT_DIR / f"global_v{GLOBAL_MODEL_VERSION}.ckpt")
        # 지나치게 노화된 업데이트는 0 가중치
        if alpha_eff <= 0.0:
            _log(f"[AGG] ignore stale/low-weight update: s={s}, alpha_eff={alpha_eff:.4f} (base_gv={base_ver})")
            ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = CKPT_DIR / f"global_v{GLOBAL_MODEL_VERSION}_{ts}.ckpt"
            # 최신 상태를 '새 파일명'으로도 남겨두면, 호출자 관점에서 경로가 항상 갱신됨
            try:
                torch.save(GLOBAL_MODEL_STATE, out_path)
            except Exception:
                pass
            return str(out_path)
        
        GLOBAL_MODEL_STATE = aggregate_params(GLOBAL_MODEL_STATE, local_state, alpha_eff)
        GLOBAL_MODEL_VERSION += 1

        ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = CKPT_DIR / f"global_v{GLOBAL_MODEL_VERSION}_{ts}.ckpt"
        torch.save(GLOBAL_MODEL_STATE, out_path)
        # canonical link-like 파일
        link_path = CKPT_DIR / f"global_v{GLOBAL_MODEL_VERSION}.ckpt"
        try:
            torch.save(GLOBAL_MODEL_STATE, link_path)
        except Exception:
            pass

        _save_last_global_ptr(out_path, GLOBAL_MODEL_VERSION)

        _log(f"[AGG] merged s={s}, tau={STALENESS_TAU}, mode={STALENESS_MODE}, "
            f"alpha_eff={alpha_eff:.4f} (base_gv={base_ver})")

    # ---- 글로벌 평가 및 메트릭 로깅 (주기 EVAL_EVERY_N) ----
    try:
        if GLOBAL_MODEL_VERSION % max(1, EVAL_EVERY_N) == 0:
            ds_val  = _get_eval_dataset("val")
            ds_test = _get_eval_dataset("test")

            calib_ds = ds_val or ds_test
            if calib_ds is None:
                _log(f"[AGG] Global v{GLOBAL_MODEL_VERSION}: skipped eval (no dataset)")

            else:
                # 1) BN 재보정 1회 (val 우선, 없으면 test)
                GLOBAL_MODEL_STATE = _bn_recalibrate(
                    GLOBAL_MODEL_STATE, calib_ds,
                    batches=int(os.getenv("FL_BN_CALIB_BATCHES","20")),
                    bs=int(os.getenv("FL_EVAL_BS","256"))
                )
                # 2) 재보정 상태를 같은 ckpt에 반영 (재현성/시작점 일치)
                try:
                    torch.save(GLOBAL_MODEL_STATE, out_path)
                    torch.save(GLOBAL_MODEL_STATE, link_path)
                except Exception:
                    pass

                _save_last_global_ptr(out_path, GLOBAL_MODEL_VERSION)

                # 3) split 별 별도 평가 (재보정은 유지하되 재보정 반복 X)
                def _eval_and_log(split, ds):
                    if ds is None: 
                        _log(f"[AGG] Global v{GLOBAL_MODEL_VERSION} {split}: no dataset")
                        return
                    with EVAL_DONE_LOCK:
                        key = (GLOBAL_MODEL_VERSION, split)
                        if key in EVAL_DONE:
                            return
                    g_loss, g_acc, n, g_f1, g_madds, g_flops, g_lat = _evaluate_state_dict(
                        GLOBAL_MODEL_STATE, ds, batch_size=EVAL_BS, device="cpu"
                    )
                    _log(f"[AGG] Global v{GLOBAL_MODEL_VERSION} {split}: acc={g_acc:.2f}% loss={g_loss:.4f} (n={n})")
                    _log_global_metrics(
                        GLOBAL_MODEL_VERSION, split, g_loss, g_acc, n, str(out_path),
                        f1_macro=g_f1, madds_M=g_madds, flops_M=g_flops, latency_ms=g_lat
                    )
                    with EVAL_DONE_LOCK:
                        EVAL_DONE.add((GLOBAL_MODEL_VERSION, split))

                _eval_and_log("val",  ds_val)
                _eval_and_log("test", ds_test)
    except Exception as e:
        logger.warning(f"[AGG] global evaluation failed: {e}")

    return str(out_path)

# === 로컬 학습 함수: do_local_training ===
def do_local_training(
    sat_id: int,
    stop_event: threading.Event,
    gpu_id: Optional[int] = None,
    *,
    epochs: Optional[int] = None,
    lr: Optional[float] = None,
    batch_size: Optional[int] = None,
) -> str:
    """
    로컬 학습을 수행하고 마지막 체크포인트 파일 경로를 반환.
    - stop_event가 set되면 가능한 빠르게 중단.
    - gpu_id가 주어지면 해당 GPU를 사용.
    - 데이터는 data.get_training_dataset(sat_id)에서 획득.
    - 글로벌 초기 가중치 훅(get_initial_model_state)이 있으면 적용.
    """
    from torch.utils.data import DataLoader
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torchvision.models import mobilenet_v3_small

    # 수치 안정성/재현성
    try:
        import torch.backends.cudnn as cudnn; cudnn.benchmark = True
    except Exception: pass
    # ---- 하이퍼파라미터: 환경변수 -> 인자 -> 기본값 ----
    EPOCHS = int(os.getenv("FL_EPOCHS_PER_ROUND", "10")) if epochs is None else int(epochs)
    LR     = float(os.getenv("FL_LR", "1e-3"))          if lr is None else float(lr)
    BS     = int(os.getenv("FL_BATCH_SIZE", "128"))      if batch_size is None else int(batch_size)
    NUM_CLASSES = int(os.getenv("FL_NUM_CLASSES", "10"))

    # ---- 디바이스 선정 ----
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_id}") if gpu_id is not None else torch.device("cuda")
    else:
        device = torch.device("cpu")

    # ---- 모델 준비 ----
    model = mobilenet_v3_small(num_classes=NUM_CLASSES)
    start_gver = globals().get("GLOBAL_MODEL_VERSION", -1)  # <- 기본값
    resumed = False
    try:
        meta = _load_last_local_ptr(sat_id)
        if meta and meta.get("path"):
            sd = torch.load(meta["path"], map_location="cpu")
            model.load_state_dict(sd, strict=False)
            # from_gver이 포인터에 없으면 파일명에서 파싱
            fg = meta.get("from_gver")
            if fg is None or fg < 0:
                fg = _parse_fromg(os.path.basename(meta["path"])) or start_gver
            start_gver = int(fg)
            resumed = True
            _log(f"SAT{sat_id}: resumed from local ckpt {meta['path']} (from_gv={start_gver})")
    except Exception as e:
        _log(f"SAT{sat_id}: failed to resume local ckpt; fallback to global. err={e}")

    if not resumed and "get_global_model_snapshot" in globals():
        try:
            start_gver, state_dict = get_global_model_snapshot()
            if state_dict is not None:
                model.load_state_dict(state_dict, strict=False)
        except Exception as e:
            _log(f"SAT{sat_id}: failed to load global snapshot v{start_gver}: {e}")
    
    _log(f"SAT{sat_id}: starting local training using global v{start_gver}")
    
    model.to(device)
    model.train()

    # ---- 데이터 준비 (반드시 data 레지스트리 사용) ----
    dataset = get_training_dataset(sat_id)
    pin_mem = torch.cuda.is_available() and DL_WORKERS > 0
    loader = DataLoader(
        dataset,
        batch_size=BS,
        # 드물게 CPU 텐서 남는 일 방지: GPU일 때만 pin_memory
        shuffle=True,
        drop_last=False,
        num_workers=DL_WORKERS,
        pin_memory=pin_mem,
        persistent_workers=(DL_WORKERS > 0),
    )

    # ---- 옵티마이저/로스 ----
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    clip_norm = float(os.getenv("FL_CLIP_NORM", "1.0"))
    max_bad_skips = int(os.getenv("FL_MAX_BAD_BATCH_SKIPS", "20"))

    def save_ckpt(ep: int) -> str:
        import datetime as _dt
        ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"sat{sat_id}_fromg{start_gver}_round{train_states[sat_id].round_idx}_ep{ep}_{ts}.ckpt"
        ckpt_path = CKPT_DIR / fname
        torch.save(model.state_dict(), ckpt_path)
        # 포인터 갱신
        _save_last_local_ptr(
            sat_id,
            str(ckpt_path),
            from_gver=start_gver,
            round_idx=train_states[sat_id].round_idx,
            epoch=ep,
        )
        # 메모리 상태에도 최신 경로 반영 (가시성 전이 시 업로드 용)
        try:
            train_states[sat_id].last_ckpt_path = str(ckpt_path)
        except Exception:
            pass
        return str(ckpt_path)

    last_ckpt = None
    n_total = len(dataset) if hasattr(dataset, "__len__") else None

    # 학습 루프
    for ep in range(EPOCHS):
        if stop_event.is_set():
            break

        running_loss, correct, total = 0.0, 0, 0
        bad_skips = 0

        # 현재 모델의 디바이스/자료형을 한 번 읽어둠
        param0 = next(model.parameters())
        device = _ensure_all_on_model_device(model, optimizer, criterion)
        model_device = param0.device
        model_dtype  = param0.dtype  # torch.float32

        for images, labels in loader:
            if stop_event.is_set():
                break

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # # --- 배치 언팩 ---
            # if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            #     images, labels = batch[0], batch[1]
            # else:
            #     images, labels = batch, None

            # --- 텐서화 & 디바이스/자료형 정합성 보장 ---
            if not torch.is_tensor(images):
                images = torch.as_tensor(images)

            # 항상 모델 파라미터의 device/dtype에 맞춘다
            # (여기서 non_blocking=True는 pinned memory일 때만 이점. pinned 아니어도 문제 없음)
            # images = images.to(device=model_device, non_blocking=True)
            assert torch.is_tensor(images), f"type(images)={type(images)}"
            if images.is_floating_point() and images.dtype != model_dtype:
                images = images.to(model_dtype)

            if labels is not None:
                if not torch.is_tensor(labels):
                    labels = torch.as_tensor(labels)
                labels = labels.to(device=model_device, non_blocking=True, dtype=torch.long)

            optimizer.zero_grad(set_to_none=True)

            # (선택) AMP 사용 시 device_type을 자동 지정
            use_amp = (device.type == "cuda")
            try:
                if use_amp:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        outputs = model(images)
                else:
                    outputs = model(images)
            except RuntimeError as e:
                msg = str(e)
                # 드문 혼합 디바이스/타입 에러가 남았을 때: 모든 상태를 강제 재정렬하고 1회만 재시도
                if ("Expected all tensors to be on the same device" in msg
                    or "and weight type" in msg
                    or "Expected object of device type" in msg):
                    # 모델/옵티마/크리테리언 상태를 다시 한번 강제 정렬
                    device = _ensure_all_on_model_device(model, optimizer, criterion)
                    images = images.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    # 재시도
                    if use_amp and device.type == "cuda":
                        with torch.autocast(device_type="cuda", dtype=torch.float16):
                            outputs = model(images)
                    else:
                        outputs = model(images)
                else:
                    # 다른 런타임 에러는 그대로 올림
                    raise

            if labels is None:
                # (x,y) 구조가 아닌 데이터셋 방어
                continue

            loss = criterion(outputs, labels)

            if not torch.isfinite(loss):
                bad_skips += 1
                if bad_skips % 5 == 0:
                    # 불안정 시 LR 다운
                    for g in optimizer.param_groups:
                        g['lr'] = max(g['lr'] * 0.5, 1e-6)
                    _log(f"SAT{sat_id}: non-finite loss; halved LR to {optimizer.param_groups[0]['lr']:.2e}")
                if bad_skips >= max_bad_skips:
                    _log(f"SAT{sat_id}: too many bad batches (>= {max_bad_skips}); breaking epoch")
                    break
                continue

            loss.backward()
            if clip_norm > 0:
                try:
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
                except Exception:
                    pass
            optimizer.step()

            running_loss += float(loss.item())
            _, pred = outputs.max(1)
            total += labels.size(0)
            correct += int(pred.eq(labels).sum().item())

        avg_loss = running_loss / max(1, len(loader))
        acc = 100.0 * correct / max(1, total)
        last_ckpt = save_ckpt(ep)
        _log(f"SAT{sat_id}: ep={ep+1}/{EPOCHS} loss={avg_loss:.4f} acc={acc:.2f}% from_gv={start_gver} saved={last_ckpt}")

        try:
            _log_local_metrics(
                sat_id=sat_id,
                round_idx=train_states[sat_id].round_idx,
                epoch=ep,
                loss=avg_loss,
                acc=acc,
                n=n_total or total,
                ckpt_path=last_ckpt
            )
        except Exception as e:
            logger.warning(f"[METRICS] local write failed (sat{sat_id}): {e}")

    return last_ckpt or save_ckpt(-1)
    
def get_global_model_snapshot():
    """(version, deepcopy(state_dict))를 원자적으로 가져온다."""
    with GLOBAL_MODEL_LOCK:
        ver = GLOBAL_MODEL_VERSION
        state = copy.deepcopy(GLOBAL_MODEL_STATE)
    return ver, state

def consolidate_existing_global_metrics(remove_old: bool = False):
    """
    과거 per-version CSV들을 단일 global_metrics.csv로 병합.
    - 이미 단일 파일이 있으면 그대로 append(중복 방지 X → 필요시 수동 정리)
    - older 파일 헤더가 (loss, acc, ckpt_path)만 있어도 자동 매핑, 나머진 빈칸
    - remove_old=True면 병합 후 기존 파일 삭제
    """
    import csv
    merged = 0
    for p in sorted(GLOBAL_METRICS_DIR.glob("global*.csv")):
        if p.name == GLOBAL_METRICS_CSV.name:
            continue
        try:
            with p.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                # 행마다 단일 헤더로 매핑
                for r in reader:
                    row = [
                        r.get("timestamp", ""),
                        r.get("version", ""),
                        r.get("split", ""),
                        r.get("num_samples", ""),
                        r.get("loss", ""),
                        r.get("acc", ""),
                        r.get("f1_macro", ""),
                        r.get("madds_M", ""),
                        r.get("flops_M", ""),
                        r.get("latency_ms", ""),
                        r.get("ckpt_path", ""),
                    ]
                    _append_csv_row(GLOBAL_METRICS_CSV, GLOBAL_METRICS_HEADER, row)
                    try:
                        _append_excel_row(GLOBAL_METRICS_XLSX, GLOBAL_METRICS_HEADER, row, sheet="global")
                    except Exception:
                        pass
                    merged += 1
            if remove_old:
                try:
                    p.unlink()
                except Exception:
                    pass
        except Exception as e:
            logger.warning(f"[METRICS] consolidate skip {p}: {e}")
    _log(f"[METRICS] consolidated {merged} rows into {GLOBAL_METRICS_CSV}")

# ==================== FastAPI 앱/수명주기 ====================
@asynccontextmanager
async def lifespan(app: FastAPI):
    await initialize_simulation()
    try:
        yield
    finally:
        # executors 정리
        training_executor.shutdown(wait=False, cancel_futures=True)
        uploader_executor.shutdown(wait=False, cancel_futures=True)


app = FastAPI(
    title="Satellite Communication API",
    description="위성, 지상국, IoT 클러스터 간 통신 상태 및 관측 가능 시간 등을 제공하는 API 서비스입니다.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)


# ==================== 서버 기동 시 처리 ====================
async def initialize_simulation():
    global satellites

    tle_path = "../constellation.tle"
    if not os.path.exists(tle_path):
        raise FileNotFoundError(f"TLE 파일이 존재하지 않습니다: {tle_path}")

    with open(tle_path, "r") as f:
        lines = [line.strip() for line in f.readlines()]
        for i in range(0, len(lines), 3):
            name, line1, line2 = lines[i:i+3]
            sat_id = int(name.replace("SAT", ""))
            satellite = EarthSatellite(line1, line2, name, ts)
            satellites[sat_id] = satellite

    # 초기 위치 계산 & FL 상태 초기화
    t = get_current_time_utc()
    for sat_id, satellite in satellites.items():
        subpoint = satellite.at(t).subpoint()
        current_sat_positions[sat_id] = {
            "lat": subpoint.latitude.degrees,
            "lon": subpoint.longitude.degrees,
        }
        sat_comm_status[sat_id] = False
        train_states[sat_id] = TrainState()
    
    # 각 위성별 로컬 포인터 복원
    try:
        for _sid in satellites.keys():
            meta = _load_last_local_ptr(_sid)
            if meta and meta.get("path"):
                train_states[_sid].last_ckpt_path = meta["path"]
                _log(f"[INIT] SAT{_sid}: restored last local ckpt -> {meta['path']} (from_gv={meta.get('from_gver')})")
    except Exception as e:
        logger.warning(f"[INIT] restore local ptr failed: {_exc_repr(e)}")

    # --- CIFAR-10 로드 & 위성별 가상 데이터셋 배정 ---
    try:
        DATA_REGISTRY.load_base(root=CIFAR_ROOT, seed=RNG_SEED, download=True)
        assign_file = ASSIGNMENTS_DIR / f"assign_seed{RNG_SEED}_alpha{DIRICHLET_ALPHA}_spc{SAMPLES_PER_CLIENT}.npz"
        if not DATA_REGISTRY.load_assignments(assign_file):
            sat_ids_sorted = sorted(satellites.keys())
            DATA_REGISTRY.assign_clients(
                sat_ids=sat_ids_sorted,
                samples_per_client=SAMPLES_PER_CLIENT,
                alpha=DIRICHLET_ALPHA,
                seed=RNG_SEED,
                with_replacement=WITH_REPLACEMENT,
            )
            DATA_REGISTRY.save_assignments(assign_file)
        logger.info(f"[DATA] CIFAR ready. per_client={SAMPLES_PER_CLIENT}, alpha={DIRICHLET_ALPHA}, seed={RNG_SEED}")
    except Exception as e:
        logger.error(f"[DATA] CIFAR init failed: {e}")

    _init_global_model()

    for gid in _build_worker_gpu_list():
        asyncio.create_task(_train_worker(gid))

    asyncio.create_task(simulation_loop())


# ==================== 시뮬레이션 루프 ====================
async def simulation_loop():
    global sim_time, current_sat_positions

    while True:
        if not sim_paused:
            t_ts = get_current_time_utc()
            current_sat_positions = {}

            for sat_id, satellite in satellites.items():
                # 이전 가시성
                prev_visible = bool(sat_comm_status.get(sat_id, False))

                # 현재 가시성 계산
                alt_deg = elevation_deg(satellite, observer, t_ts)
                visible_now = (alt_deg >= threshold_deg)
                sat_comm_status[sat_id] = visible_now

                # 위치 업데이트
                subpoint = satellite.at(t_ts).subpoint()
                current_sat_positions[sat_id] = {
                    "lat": subpoint.latitude.degrees,
                    "lon": subpoint.longitude.degrees,
                }

                # 연합학습 트리거
                if not visible_now:
                    # 오프라인 → 학습 시작(가능 시)
                    _enqueue_training(sat_id)
                elif (not prev_visible) and visible_now:
                    # False -> True 전이
                    _on_become_visible(sat_id)
                    train_states[sat_id].round_idx += 1

            sim_time += timedelta(seconds=sim_delta_sec)

        await asyncio.sleep(real_interval_sec)


# ==================== 대시보드 / 페이지 ====================
@app.get("/dashboard", response_class=HTMLResponse, tags=["PAGE"])
def dashboard():
    """
    대시보드 HTML 페이지
    """
    paused_status = "Paused" if sim_paused else "Running"
    return f"""
    <html>
    <head>
        <title>Satellite Communication Dashboard</title>
        <meta http-equiv="refresh" content="5">
        <style>
            body {{ font-family: Arial; background: #f2f2f2; margin: 2em; }}
            h1 {{ color: #333; }}
            a {{ display: block; margin: 10px 0; font-size: 1.2em; }}
        </style>
    </head>
    <body>
        <h1>🛰️🛰 Satellite Communication Dashboard</h1>
        <p><b>Sim Time:</b> {sim_time.isoformat()}</p>
        <p><b>Status:</b> {paused_status}</p>
        <p><b>Step:</b> Δt={sim_delta_sec}s, Interval={real_interval_sec}s</p>
        <div class="control-form">
            <label>Δt(초): <input id="delta" type="number" step="any" value="{sim_delta_sec}" /></label>
            <label>간격(초): <input id="interval" type="number" step="any" value="{real_interval_sec}" /></label>
            <button onclick="setStep()">적용</button>
        </div>
        <script>
        async function setStep() {{
            const d = document.getElementById('delta').value;
            const i = document.getElementById('interval').value;
            const res = await fetch(`/api/set_step?delta_sec=${{d}}&interval_sec=${{i}}`, {{ method: 'PUT' }});
            const data = await res.json();
            if (!data.error) {{
                alert(`설정 완료: Δt=${{data.sim_delta_sec}}, Interval=${{data.real_interval_sec}}`);
                window.location.reload();
            }} else {{
                alert(`오류: ${{data.error}}`);
            }}
        }}
        </script>
        <hr>
        <a href="/gs_visibility">🛰️ GS별 통신 가능 위성 보기</a>
        <a href="/orbit_paths/lists">🛰 위성별 궤적 경로 보기</a>
        <a href="/map_path">🗺 지도 기반 위성 경로 보기</a>
        <a href="/visibility_schedules/lists">📅 위성별 관측 가능 시간대 목록 보기</a>
        <a href="/iot_clusters"> 📡 IoT 클러스터별 위치 보기</a>
        <a href="/iot_visibility"> 🌐 IoT 클러스터별 통신 가능 위성 보기</a>
        <a href="/comm_targets/lists">🚀 위성 통신 대상 확인</a>
    </body>
    </html>
    """


@app.get("/gs_visibility", response_class=HTMLResponse, tags=["PAGE"])
def gs_visibility():
    """
    지상국별로 관측 가능한 위성 목록을 HTML로 반환하는 페이지
    """

    paused_status = "Paused" if sim_paused else "Running"
    gs_sections = []
    t_ts = get_current_time_utc()

    for name, gs in observer_locations.items():
        rows = []
        for sid, sat in satellites.items():
            alt_deg = elevation_deg(sat, gs, t_ts)
            if alt_deg >= threshold_deg:
                rows.append(f'<tr><td>{sid}</td><td>{alt_deg:.2f}°</td></tr>')
        table_html = f"""
        <h2>{name}</h2>
        <table>
            <tr><th>Sat ID</th><th>Elevation</th></tr>
            {''.join(rows)}
        </table>
        """
        gs_sections.append(table_html)

    return f"""
    <html>
    <head>
        <title>GS Visibility</title>
        <meta http-equiv="refresh" content="5">
        <style>
            body {{ font-family: Arial; background: #f2f2f2; margin: 2em; }}
            h2 {{ margin-top: 2em; }}
            table {{ border-collapse: collapse; width: 60%; margin-bottom: 2em; }}
            th, td {{ border: 1px solid #ccc; padding: 8px; text-align: center; }}
        </style>
    </head>
    <body>
        <p><a href="/dashboard">← Back to Dashboard</a></p>
        <h1>🛰️ GS-wise Visible Satellites</h1>
        <p><b>Sim Time:</b> {sim_time.isoformat()}</p>
        <p><b>Status:</b> {paused_status}</p>
        <p><b>Step:</b> Δt={sim_delta_sec}s, Interval={real_interval_sec}s</p>
        <hr>
        {''.join(gs_sections)}
    </body>
    </html>
    """


@app.get("/orbit_paths/lists", response_class=HTMLResponse, tags=["PAGE"])
def sat_paths():
    """
    위성별 궤적 경로 링크 목록을 HTML로 반환하는 페이지
    """
    links = [f'<li><a href="/orbit_paths?sat_id={sid}">SAT{sid} Path</a></li>' for sid in sorted(satellites)]
    return f"""
    <html>
    <head>
        <title>Satellite Paths</title>
        <style>
            body {{ font-family: Arial; margin: 2em; }}
            ul {{ list-style-type: none; padding: 0; }}
            li {{ margin-bottom: 10px; }}
        </style>
    </head>
    <body>
        <p><a href="/dashboard">← Back to Dashboard</a></p>
        <h1>🛰 All Satellite Orbit Paths</h1>
        <ul>
            {''.join(links)}
        </ul>
    </body>
    </html>
    """


@app.get("/orbit_paths", response_class=HTMLResponse, tags=["PAGE"])
def orbit_paths(sat_id: int = Query(...)):
    """
    특정 위성의 궤적 경로를 HTML로 반환하는 페이지
    """
    if sat_id not in satellites:
        return HTMLResponse(f"<p>Error: sat_id {sat_id} not found</p>", status_code=404)

    satellite = satellites[sat_id]
    t0 = sim_time
    positions = []
    for offset_sec in range(0, 7200, 60):
        future = t0 + timedelta(seconds=offset_sec)
        t_ts = to_ts(future)
        subpoint = satellite.at(t_ts).subpoint()
        positions.append((subpoint.latitude.degrees, subpoint.longitude.degrees))

    rows = ''.join([f'<tr><td>{i*60}s</td><td>{lat:.2f}</td><td>{lon:.2f}</td></tr>'
                    for i, (lat, lon) in enumerate(positions)])

    return f"""
    <html>
    <head>
        <title>Orbit Track for SAT{sat_id}</title>
        <style>
            body {{ font-family: Arial; margin: 2em; }}
            table {{ border-collapse: collapse; width: 60%; }}
            th, td {{ border: 1px solid #ccc; padding: 8px; text-align: center; }}
        </style>
    </head>
    <body>
        <p><a href="/orbit_paths/lists">← Back to All Satellite Orbit Paths</a></p>
        <h1>🛰 SAT{sat_id} Orbit Path</h1>
        <table>
            <tr><th>Offset</th><th>Latitude</th><th>Longitude</th></tr>
            {rows}
        </table>
    </body>
    </html>
    """


@app.get("/map_path", response_class=HTMLResponse, tags=["PAGE"])
def map_path():
    """
    지도 기반 위성 경로를 표시하는 HTML 페이지
    """
    options = ''.join(f'<option value="{sid}">SAT{sid}</option>' for sid in sorted(satellites))
    obs_data_json = json.dumps({name: {"lat": gs.latitude.degrees, "lon": gs.longitude.degrees}
                                for name, gs in observer_locations.items()})
    iot_data_json = json.dumps(raw_iot_clusters)
    return f"""
    <html>
    <head>
        <title>Map Path</title>
        <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
        <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
        <style>
            #map {{ height: 90vh; }}
            html, body {{ margin: 0; padding: 0; }}
            .coverage-circle {{ fill-opacity:0.5; stroke-width:0; }}
            .leaflet-tooltip.no-box {{
                background: transparent; border: none; box-shadow: none; padding: 0; font-weight: bold;
            }}
            .leaflet-tooltip.no-box::before {{ display: none; }}
        </style>
    </head>
    <body>
        <p><a href="/dashboard">← Back to Dashboard</a></p>
        <h1>🗺 Satellite Map Path</h1>
        <div id="sim-time"></div>
        <div id="sim-step" style="margin-bottom: 1em;"></div>
        <label for="sat_id">Choose a satellite:</label>
        <select id="sat_id" onchange="drawTrajectory(this.value)">{options}</select>
        <div id="map"></div>
        <script>
            var map = L.map('map', {{
                center: [0, 0],
                zoom: 2,
                worldCopyJump: false,
                maxBounds: [[-85, -180], [85, 180]],
                maxBoundsViscosity: 1.0,
                inertia: false
            }});
            L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png').addTo(map);

            const observers = {obs_data_json};
            const iotClusters = {iot_data_json};

            var circles = [];
            var pathLines = [];
            var currentMarker = null;
            var currentLabel = null;
            var markerInterval = null;
            var coverCircles = [];

            function drawCoverage() {{
                for (let [name, loc] of Object.entries(observers)) {{
                    let c = L.circle([loc.lat, loc.lon], {{
                        radius: 100000, color: 'green', fillColor: 'green', className: 'coverage-circle'
                    }}).addTo(map);
                    c.bindTooltip(name, {{ permanent: true, direction: 'center', className: 'no-box' }});
                    coverCircles.push(c);
                }}
                for (let [name, loc] of Object.entries(iotClusters)) {{
                    let c = L.circle([loc.latitude, loc.longitude], {{
                        radius: 50000, color: 'orange', fillColor: 'orange', className: 'coverage-circle'
                    }}).addTo(map);
                    c.bindTooltip(name, {{ permanent: true, direction: 'center', className: 'no-box' }});
                    coverCircles.push(c);
                }}
            }}

            async function drawTrajectory(sat_id) {{
                circles.forEach(c => map.removeLayer(c)); circles = [];
                pathLines.forEach(p => map.removeLayer(p)); pathLines = [];
                if (currentMarker) map.removeLayer(currentMarker);
                if (currentLabel) map.removeLayer(currentLabel);
                if (markerInterval) clearInterval(markerInterval);

                drawCoverage();

                let resp = await fetch(`/api/trajectory?sat_id=${{sat_id}}`);
                let satData = await resp.json();
                for (let segment of satData.segments) {{
                    let latlngs = [];
                    for (let p of segment) {{
                        let latlng = [p.lat, p.lon];
                        latlngs.push(latlng);
                        let marker = L.circleMarker(latlng, {{radius: 1, color: 'red'}}).addTo(map);
                        circles.push(marker);
                    }}
                    let line = L.polyline(latlngs, {{color: 'blue', weight: 1}}).addTo(map);
                    pathLines.push(line);
                }}
                markerInterval = setInterval(async () => {{
                    let live = await fetch(`/api/position?sat_id=${{sat_id}}`);
                    let data = await live.json();
                    let simResp = await fetch(`/api/sim_time`);
                    let simData = await simResp.json();
                    document.getElementById('sim-time').innerHTML = `<p><b>Sim Time:</b> ${{simData.sim_time}}</p>`;
                    document.getElementById('sim-step').innerHTML = `<p><b>Step:</b> Δt={sim_delta_sec}s, Interval={real_interval_sec}s</p>`;

                    if (data.lat !== undefined && data.lon !== undefined) {{
                        if (currentMarker) map.removeLayer(currentMarker);
                        if (currentLabel) map.removeLayer(currentLabel);
                        currentMarker = L.circleMarker([data.lat, data.lon], {{radius: 3, color: 'blue'}}).addTo(map);
                        currentLabel = L.marker([data.lat, data.lon], {{
                            icon: L.divIcon({{
                                className: 'current-label',
                                html: '<b>현재 위성 위치</b>',
                                iconSize: [120, 20],
                                iconAnchor: [60, -10]
                            }})
                        }}).addTo(map);
                    }}
                }}, 1000);
            }}

            window.onload = () => {{
                const selector = document.getElementById('sat_id');
                if (selector.value) drawTrajectory(selector.value);
            }}
        </script>
    </body>
    </html>
    """

@app.get("/visibility_schedules/lists", response_class=HTMLResponse, tags=["PAGE"])
def get_list_visibility_schedules():
    """
    위성별 관측 가능 시간대 링크 목록을 HTML로 반환하는 페이지
    """
    links = [f'<li><a href="/visibility_schedules?sat_id={sid}">SAT{sid} Schedule</a></li>' for sid in sorted(satellites)]
    return f"""
    <html>
    <head>
        <title>Visibility Schedule List</title>
        <style>
            body {{ font-family: Arial; margin: 2em; }}
            ul {{ list-style-type: none; padding: 0; }}
            li {{ margin-bottom: 10px; }}
        </style>
    </head>
    <body>
        <p><a href="/dashboard">← Back to Dashboard</a></p>
        <h1>📅 All Satellite Visibility Schedules</h1>
        <ul>
            {''.join(links)}
        </ul>
    </body>
    </html>
    """

@app.get("/visibility_schedules", response_class=HTMLResponse, tags=["PAGE"])
def visibility_schedules(sat_id: int = Query(...)):
    """
    특정 위성의 관측 가능 시간대를 HTML로 반환하는 페이지
    """
    if sat_id not in satellites:
        return HTMLResponse(f"<p>Error: sat_id {sat_id} not found</p>", status_code=404)

    satellite = satellites[sat_id]
    results = []
    for name, gs in observer_locations.items():
        visible_periods = []
        visible = False
        start = None
        for offset in range(0, 7200, 30):
            future = sim_time + timedelta(seconds=offset)
            t = ts.utc(future.year, future.month, future.day, future.hour, future.minute, future.second)
            difference = satellite - gs
            topocentric = difference.at(t)
            alt, _, _ = topocentric.altaz()
            if alt.degrees >= threshold_deg:
                if not visible:
                    start = future
                    visible = True
            else:
                if visible:
                    visible_periods.append((start, future))
                    visible = False
        if visible and start:
            visible_periods.append((start, future))
        results.append((name, visible_periods))

    sections = []
    for name, periods in results:
        rows = ''.join(f"<tr><td>{start.strftime('%H:%M:%S')}</td><td>{end.strftime('%H:%M:%S')}</td></tr>" for start, end in periods)
        sections.append(f"<h2>{name}</h2><table><tr><th>Start</th><th>End</th></tr>{rows}</table>")

    return f"""
    <html>
    <head>
        <title>Visibility Schedule</title>
        <meta http-equiv="refresh" content="5">
        <style>
            body {{ font-family: Arial; margin: 2em; }}
            table {{ border-collapse: collapse; width: 60%; margin-bottom: 2em; }}
            th, td {{ border: 1px solid #ccc; padding: 8px; text-align: center; }}
        </style>
    </head>
    <body>
        <p><a href="/visibility_schedules/lists">← Back to Satellite Visibility Schedule List</a></p>
        <h1>📅 Visibility Schedule for SAT{sat_id}</h1>
        <p><b>Sim Time:</b> {sim_time.isoformat()}</p>
        {''.join(sections)}
    </body>
    </html>
    """

@app.get("/iot_clusters", response_class=HTMLResponse, tags=["PAGE"])
def iot_clusters_ui():
    """
    IoT 클러스터 위치를 HTML로 반환하는 페이지
    """
    rows = []
    for name, loc in raw_iot_clusters.items():
        # console.log(f"Adding IoT cluster: {name} at {loc['latitude']}, {loc['longitude']}")
        rows.append(f"<tr><td>{name}</td><td>{loc['latitude']:.2f}</td><td>{loc['longitude']:.2f}</td></tr>")
    return f"""
    <html>
    <head>
        <title>IoT Clusters</title>
        <link rel=\"stylesheet\" href=\"https://unpkg.com/leaflet@1.9.4/dist/leaflet.css\" />
        <script src=\"https://unpkg.com/leaflet@1.9.4/dist/leaflet.js\"></script>
        <style>
            body {{ font-family: Arial; margin: 2em; }}
            table {{ border-collapse: collapse; width: 60%; }}
            th, td {{ border: 1px solid #ccc; padding: 8px; text-align: center; }}
            #map {{ height: 500px; margin-top: 30px; }}
        </style>
    </head>
    <body>
        <p><a href="/dashboard">← Back to Dashboard</a></p>
        <h1>📡 IoT Cluster Locations</h1>
        <table>
            <tr><th>Name</th><th>Latitude</th><th>Longitude</th></tr>
            {''.join(rows)}
        </table>
        <hr/>
        <div id=\"map\"></div>
        <script>
            var map = L.map('map', {{
                center: [0, 0],
                zoom: 2,
                worldCopyJump: false,
                maxBounds: [[-85, -180], [85, 180]],
                maxBoundsViscosity: 1.0,
                inertia: false
            }});
            L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
                maxZoom: 18,
                attribution: '© OpenStreetMap contributors'
            }}).addTo(map);

            const clusters = {raw_iot_clusters};
            console.log("IoT clusters:", clusters);
            for (let [name, loc] of Object.entries(clusters)) {{
                const lat = loc.latitude;
                const lon = loc.longitude;
                const marker = L.circleMarker([lat, lon], {{ radius: 5, color: 'blue' }}).addTo(map);
                marker.bindTooltip(name, {{
                    permanent: true,
                    direction: 'top',
                    className: 'iot-tooltip'
                }});
            }}
        </script>
    </body>
    </html>
    """

@app.get("/iot_visibility", response_class=HTMLResponse, tags=["PAGE"])
def iot_visibility():
    """
    IoT 클러스터에서 관측 가능한 위성 목록을 HTML로 반환하는 페이지
    """
    iot_sections = []
    for name, iot in iot_clusters.items():
        t = ts.utc(sim_time.year, sim_time.month, sim_time.day, sim_time.hour, sim_time.minute, sim_time.second)
        rows = []
        for sid, sat in satellites.items():
            difference = sat - iot
            topocentric = difference.at(t)
            alt, az, dist = topocentric.altaz()
            if alt.degrees >= threshold_deg:
                rows.append(f'<tr><td>{sid}</td><td>{alt.degrees:.2f}°</td></tr>')
        table_html = f"""
        <h2>{name}</h2>
        <table>
            <tr><th>Sat ID</th><th>Elevation</th></tr>
            {''.join(rows)}
        </table>
        """
        iot_sections.append(table_html)

    return f"""
    <html>
    <head>
        <title>IOT Visibility</title>
        <meta http-equiv="refresh" content="5">
        <style>
            body {{ font-family: Arial; background: #f2f2f2; margin: 2em; }}
            h2 {{ margin-top: 2em; }}
            table {{ border-collapse: collapse; width: 60%; margin-bottom: 2em; }}
            th, td {{ border: 1px solid #ccc; padding: 8px; text-align: center; }}
        </style>
    </head>
    <body>
        <p><a href="/dashboard">← Back to Dashboard</a></p>
        <h1>🌐 IOT-wise Visible Satellites</h1>
        <p><b>Sim Time:</b> {sim_time.isoformat()}</p>
        <hr>
        {''.join(iot_sections)}
    </body>
    </html>
    """

@app.get("/comm_targets/lists", response_class=HTMLResponse, tags=["PAGE"])  
def comm_targets_list():
    """
    위성 ID 목록을 보여주고 각 위성의 통신 대상을 상세 페이지로 연결하는 리스트 페이지
    """
    links = [f'<li><a href="/comm_targets?sat_id={sid}">SAT{sid} Targets</a></li>' for sid in sorted(satellites)]
    return f"""
    <html>
    <head>
        <title>Comm Targets List</title>
        <meta http-equiv="refresh" content="5">
        <style>
            body {{ font-family: Arial; margin: 2em; }}
            ul {{ list-style-type: none; padding: 0; }}
            li {{ margin-bottom: 5px; }}
        </style>
    </head>
    <body>
        <p><a href="/dashboard">← Back to Dashboard</a></p>
        <h1>🚀 Satellite Comm Targets</h1>
        <ul>
            {''.join(links)}
        </ul>
    </body>
    </html>
    """

@app.get("/comm_targets", response_class=HTMLResponse, tags=["PAGE"])  
def comm_targets_detail(sat_id: int = Query(..., description="위성 ID")):
    """
    특정 위성의 현재 통신 가능한 지상국과 IoT 클러스터를 HTML 테이블로 보여주는 상세 페이지
    """
    if sat_id not in satellites:
        return HTMLResponse(f"<p style='color:red;'>Error: sat_id {sat_id} not found</p>", status_code=404)
    data = get_comm_targets(sat_id)
    # 테이블 생성
    rows_ground = ''.join(f"<tr><td>{gs}</td></tr>" for gs in data['visible_ground_stations']) or '<tr><td>None</td></tr>'
    rows_iot    = ''.join(f"<tr><td>{ci}</td></tr>" for ci in data['visible_iot_clusters']) or '<tr><td>None</td></tr>'
    return f"""
    <html>
    <head>
        <title>Comm Targets for SAT{{sat_id}}</title>
        <meta http-equiv="refresh" content="5">
        <style>
            body {{ font-family: Arial; margin: 2em; }}
            table {{ border-collapse: collapse; width: 40%; margin-top: 1em; }}
            th, td {{ border: 1px solid #ccc; padding: 8px; text-align: left; }}
        </style>
    </head>
    <body>
        <p><a href="/comm_targets/lists">← Back to Comm Targets List</a></p>
        <h1>📡 Comm Targets for SAT{sat_id}</h1>
        <p><b>Sim Time:</b> {data['sim_time']}</p>
        <h2>Visible Ground Stations</h2>
        <table><tr><th>Station</th></tr>{rows_ground}</table>
        <h2>Visible IoT Clusters</h2>
        <table><tr><th>Cluster</th></tr>{rows_iot}</table>
    </body>
    </html>
    """

# ==================== API ====================
@app.post("/api/reset_time", tags=["API"])
def reset_sim_time():
    """
    시뮬레이션 시간을 초기화하는 API
    """
    global sim_time
    sim_time = datetime(2025, 3, 30, 0, 0, 0)
    return {"status": "reset", "sim_time": sim_time.isoformat()}


@app.get("/api/sim_time", tags=["API"])
def get_sim_time_api():
    """
    현재 시뮬레이션 시간을 반환하는 API
    """
    return {"sim_time": sim_time.isoformat()}


@app.put("/api/sim_time", tags=["API"])
def set_sim_time(year: int, month: int, day: int, hour: int = 0, minute: int = 0, second: int = 0):
    """
    시뮬레이션 시간을 설정하는 API
    """
    global sim_time
    try:
        sim_time = datetime(year, month, day, hour, minute, second)
        return {"status": "updated", "sim_time": sim_time.isoformat()}
    except Exception as e:
        return {"error": str(e)}


@app.put("/api/set_step", tags=["API"])
def set_step(delta_sec: float = Query(...), interval_sec: float = Query(...)):
    """
    시뮬레이션 델타 시간 및 루프 간 실제 대기시간을 설정
    """
    global sim_delta_sec, real_interval_sec
    if delta_sec <= 0 or interval_sec < 0:
        return {"error": "delta_sec은 양수, interval_sec은 0 이상이어야 합니다."}
    sim_delta_sec = float(delta_sec)
    real_interval_sec = float(interval_sec)
    return {"sim_delta_sec": sim_delta_sec, "real_interval_sec": real_interval_sec}


@app.post("/api/pause", tags=["API"])
def pause_simulation():
    """
    시뮬레이션을 일시정지하는 API
    """
    global sim_paused
    sim_paused = True
    return {"status": "paused"}


@app.post("/api/resume", tags=["API"])
def resume_simulation():
    """
    시뮬레이션을 재개하는 API
    """
    global sim_paused
    sim_paused = False
    return {"status": "resumed"}


@app.get("/api/trajectory", tags=["API"])
def get_trajectory(sat_id: int = Query(...)):
    """
    특정 위성의 궤적 경로를 반환하는 API
    """
    if sat_id not in satellites:
        return {"error": f"sat_id {sat_id} not found"}
    satellite = satellites[sat_id]
    t0 = sim_time
    prev_lon = None
    segment: List[Dict[str, float]] = []
    segments: List[List[Dict[str, float]]] = []

    for offset_sec in range(0, 7200, 90):
        future = t0 + timedelta(seconds=offset_sec)
        t_ts = to_ts(future)
        subpoint = satellite.at(t_ts).subpoint()
        lat = subpoint.latitude.degrees
        lon = subpoint.longitude.degrees
        if prev_lon is not None and abs(lon - prev_lon) > 180:
            segments.append(segment)
            segment = []
        segment.append({"lat": lat, "lon": lon})
        prev_lon = lon

    if segment:
        segments.append(segment)
    return {"sat_id": sat_id, "segments": segments}


@app.get("/api/position", tags=["API"])
def get_position(sat_id: int = Query(...)):
    """
    특정 위성의 현재 위치를 반환하는 API
    """
    if sat_id not in current_sat_positions:
        return {"error": f"Position for SAT{sat_id} not available"}
    return current_sat_positions[sat_id]


@app.get("/api/comm_targets", tags=["API"])
def get_comm_targets(sat_id: int = Query(..., description="위성 ID")):
    """
    주어진 위성 ID에 대해 현재 통신 가능한 지상국과 IoT 클러스터를 반환하는 API
    """
    if sat_id not in satellites:
        return {"error": f"sat_id {sat_id} not found"}
    t_ts = get_current_time_utc()
    sat = satellites[sat_id]
    visible_ground = [name for name, gs in observer_locations.items()
                      if elevation_deg(sat, gs, t_ts) >= threshold_deg]
    visible_iot = [name for name, cluster in iot_clusters.items()
                   if elevation_deg(sat, cluster, t_ts) >= threshold_deg]
    return {
        "sim_time": sim_time.isoformat(),
        "sat_id": sat_id,
        "visible_ground_stations": visible_ground,
        "visible_iot_clusters": visible_iot,
    }


@app.get("/api/gs/visibility", tags=["API/GS"])
def get_gs_visibility():
    """
    현재 시뮬레이션 시간에 각 지상국에서 관측 가능한 위성 목록을 반환하는 API
    """
    result = {}
    t_ts = get_current_time_utc()
    for name, gs in observer_locations.items():
        visible_sats = []
        for sid, sat in satellites.items():
            alt_deg = elevation_deg(sat, gs, t_ts)
            if alt_deg >= threshold_deg:
                visible_sats.append({"sat_id": sid, "elevation": alt_deg})
        result[name] = visible_sats
    return {"sim_time": sim_time.isoformat(), "data": result}


@app.get("/api/gs/visibility_schedule", tags=["API/GS"])
def get_visibility_schedule(sat_id: int = Query(...)):
    """
    특정 위성의 관측 가능 시간대를 반환하는 API
    """
    if sat_id not in satellites:
        return {"error": f"sat_id {sat_id} not found"}
    satellite = satellites[sat_id]
    results = {}

    for name, gs in observer_locations.items():
        visible_periods: List[tuple[str, str]] = []
        visible = False
        start: Optional[datetime] = None
        for offset in range(0, 7200, 30):
            future = sim_time + timedelta(seconds=offset)
            t_ts = to_ts(future)
            alt_deg = elevation_deg(satellite, gs, t_ts)
            if alt_deg >= threshold_deg:
                if not visible:
                    start = future
                    visible = True
            else:
                if visible:
                    visible_periods.append((start.isoformat(), future.isoformat()))
                    visible = False
        if visible and start:
            visible_periods.append((start.isoformat(), future.isoformat()))
        results[name] = visible_periods

    return {"sim_time": sim_time.isoformat(), "sat_id": sat_id, "schedule": results}


@app.get("/api/gs/next_comm", tags=["API/GS"])
def get_next_comm_for_sat(sat_id: int = Query(..., description="위성 ID")):
    """
    주어진 sat_id 위성이 다음에 통신 가능한 지상국과 시간(시뮬레이션 시간)을 반환하는 API
    """
    if sat_id not in satellites:
        return {"error": f"sat_id {sat_id} not found"}
    sat = satellites[sat_id]
    t0 = sim_time
    horizon = 86400
    step = max(1, int(sim_delta_sec))

    for offset in range(step, horizon + step, step):
        t_future = t0 + timedelta(seconds=offset)
        if any(elevation_deg(sat, gs, to_ts(t_future)) >= threshold_deg for gs in observer_locations.values()):
            # 가장 먼저 통신되는 지상국 이름도 함께 찾기
            for name, gs in observer_locations.items():
                if elevation_deg(sat, gs, to_ts(t_future)) >= threshold_deg:
                    return {"sat_id": sat_id, "ground_station": name, "next_comm_time": t_future.isoformat()}
    return {"sat_id": sat_id, "next_comm_time": None, "message": "다음 24시간 내 통신 창 없음"}


@app.get("/api/gs/next_comm_all", tags=["API/GS"])
def get_next_comm_all():
    """
    모든 위성에 대해 다음 통신 가능 시간(지상국, 시뮬레이션 시간)을 반환하는 API
    """
    t0 = sim_time
    horizon = 86400
    step = max(1, int(sim_delta_sec))
    result = {}

    for sat_id, sat in satellites.items():
        next_time = None
        next_gs = None
        for offset in range(step, horizon + step, step):
            t_future = t0 + timedelta(seconds=offset)
            for name, gs in observer_locations.items():
                if elevation_deg(sat, gs, to_ts(t_future)) >= threshold_deg:
                    next_time = t_future
                    next_gs = name
                    break
            if next_time:
                break
        result[sat_id] = {
            "ground_station": next_gs,
            "next_comm_time": next_time.isoformat() if next_time else None
        }
    return {"sim_time": sim_time.isoformat(), "next_comm": result}


@app.put("/api/observer", tags=["API/Observer"])
def set_observer(name: str = Query(...)):
    """
    지상국 관측 위치를 설정하는 API
    """
    global observer, current_observer_name
    if name not in observer_locations:
        return {"error": f"observer '{name}' is not supported"}
    observer = observer_locations[name]
    current_observer_name = name
    return {"status": "ok", "observer": name}


@app.get("/api/observer/check_comm", tags=["API/Observer"])
def get_observer_check_comm(sat_id: int = Query(...)):
    """
    위성 통신 가능 여부를 확인하는 API
    """
    available = bool(sat_comm_status.get(sat_id, False))
    return {"sat_id": sat_id, "observer": current_observer_name, "sim_time": sim_time.isoformat(), "available": available}


@app.get("/api/observer/all_visible", tags=["API/Observer"])
def get_all_visible():
    """
    현재 시뮬레이션 시간에 지상국(observer)에서 관측 가능한 모든 위성의 ID를 반환하는 API
    """
    visible_sats = [sat_id for sat_id, available in sat_comm_status.items() if bool(available)]
    return {"observer": current_observer_name, "sim_time": sim_time.isoformat(), "visible_sat_ids": visible_sats}


@app.get("/api/observer/visible_count", tags=["API/Observer"])
def get_visible_count():
    """
    현재 시뮬레이션 시간에 지상국(observer)에서 관측 가능한 위성의 개수를 반환하는 API
    """
    count = sum(1 for available in sat_comm_status.values() if bool(available))
    return {"observer": current_observer_name, "sim_time": sim_time.isoformat(), "visible_count": count}


@app.get("/api/observer/elevation", tags=["API/Observer"])
def get_elevation(sat_id: int = Query(...)):
    """
    특정 위성의 현재 시뮬레이션 시간에 대한 지상국과의 고도를 반환하는 API
    """
    satellite = satellites.get(sat_id)
    if satellite is None:
        return {"error": f"sat_id {sat_id} not found"}
    alt_deg = elevation_deg(satellite, observer, get_current_time_utc())
    return {"sat_id": sat_id, "observer": current_observer_name, "sim_time": sim_time.isoformat(), "elevation_deg": alt_deg}


@app.get("/api/observer/next_comm", tags=["API/Observer"])
def get_next_comm_with_observer(sat_id: int = Query(..., description="위성 ID")):
    """
    주어진 sat_id 위성이 현재 observer와 다음에 통신 가능한 시간을 반환하는 API
    """
    if sat_id not in satellites:
        return {"error": f"sat_id {sat_id} not found"}
    sat = satellites[sat_id]
    t0 = sim_time
    horizon = 86400
    step = max(1, int(sim_delta_sec))

    for offset in range(step, horizon + step, step):
        t_future = t0 + timedelta(seconds=offset)
        if elevation_deg(sat, observer, to_ts(t_future)) >= threshold_deg:
            return {"sat_id": sat_id, "observer": current_observer_name, "next_comm_time": t_future.isoformat()}
    return {"sat_id": sat_id, "observer": current_observer_name, "next_comm_time": None, "message": "다음 24시간 내 통신 창 없음"}


@app.get("/api/observer/next_comm_all", tags=["API/Observer"])
def get_next_comm_all_with_observer():
    """
    모든 위성에 대해 현재 observer 와의 다음 통신 가능한 시간을 반환하는 API
    """
    t0 = sim_time
    horizon = 86400
    step = max(1, int(sim_delta_sec))
    result = {}

    for sat_id, sat in satellites.items():
        next_time = None
        for offset in range(step, horizon + step, step):
            t_future = t0 + timedelta(seconds=offset)
            if elevation_deg(sat, observer, to_ts(t_future)) >= threshold_deg:
                next_time = t_future.isoformat()
                break
        result[sat_id] = {"next_comm_time": next_time}

    return {"observer": current_observer_name, "sim_time": sim_time.isoformat(), "next_comm": result}


@app.get("/api/iot_clusters/position", tags=["API/IoT"])
def get_iot_clusters_position():
    """
    IoT 클러스터의 위치 정보를 반환하는 API
    """
    # Topos는 직렬화 불가 → lat/lon만 반환
    clusters = {name: {"lat": t.latitude.degrees, "lon": t.longitude.degrees, "elev_m": raw_iot_clusters[name]["elevation_m"]}
                for name, t in iot_clusters.items()}
    return {"sim_time": sim_time.isoformat(), "clusters": clusters}


@app.get("/api/iot_clusters/visibility_schedule", tags=["API/IoT"])
def get_iot_clusters_visibility(sat_id: int = Query(...)):
    """
    특정 위성이 IoT 클러스터에서 관측 가능한지 여부와 관측 가능 시간대를 반환하는 API
    """
    if sat_id not in satellites:
        return {"error": f"sat_id {sat_id} not found"}
    satellite = satellites[sat_id]
    result = []

    for name, cluster in iot_clusters.items():
        visible_periods: List[tuple[str, str]] = []
        visible = False
        start: Optional[datetime] = None
        for offset in range(0, 7200, 30):
            future = sim_time + timedelta(seconds=offset)
            if elevation_deg(satellite, cluster, to_ts(future)) >= threshold_deg:
                if not visible:
                    start = future
                    visible = True
            else:
                if visible:
                    visible_periods.append((start.isoformat(), future.isoformat()))
                    visible = False
        if visible and start:
            visible_periods.append((start.isoformat(), future.isoformat()))
        if visible_periods:
            result.append({"iot_cluster": name, "periods": visible_periods})
    return {"sim_time": sim_time.isoformat(), "sat_id": sat_id, "schedule": result}


@app.get("/api/iot_clusters/visible", tags=["API/IoT"])
def get_iot_clusters_visible(
    sat_id: Optional[int] = Query(None, description="위성 ID"),
    iot_name: Optional[str] = Query(None, description="IoT 클러스터 이름")
):
    """
    특정 IoT 클러스터에서 관측 가능한지 여부를 반환하는 API\n
    - sat_id만 지정 → 해당 위성과 통신 가능한 IoT 클러스터 목록 반환
    - iot_name만 지정 → 해당 IoT 클러스터와 통신 가능한 위성 ID 목록 반환
    """
    if (sat_id is None) == (iot_name is None):
        return {"error": "neither sat_id or iot_name not found"}

    t_ts = get_current_time_utc()

    if sat_id is not None:
        if sat_id not in satellites:
            return {"error": f"sat_id {sat_id} not found"}
        satellite = satellites[sat_id]
        visible_clusters = [name for name, cluster in iot_clusters.items()
                            if elevation_deg(satellite, cluster, t_ts) >= threshold_deg]
        return {"sat_id": sat_id, "sim_time": sim_time.isoformat(), "visible_iot_clusters": visible_clusters}

    cluster_name = iot_name
    if cluster_name not in iot_clusters:
        return {"error": f"iot cluster '{cluster_name}' not found"}
    cluster = iot_clusters[cluster_name]
    visible_sats = [sid for sid, sat in satellites.items()
                    if elevation_deg(sat, cluster, t_ts) >= threshold_deg]
    return {"sim_time": sim_time.isoformat(), "iot_cluster": cluster_name, "visible_sat_ids": visible_sats}


@app.get("/api/iot_clusters/visible_count", tags=["API/IoT"])
def get_iot_clusters_visible_count(iot_name: str = Query(...)):
    """
    특정 IoT 클러스터에서 관측 가능한 위성의 개수를 반환하는 API
    """
    if iot_name not in iot_clusters:
        return {"error": f"iot cluster '{iot_name}' not found"}
    cluster = iot_clusters[iot_name]
    t_ts = get_current_time_utc()
    count = sum(1 for _, sat in satellites.items() if elevation_deg(sat, cluster, t_ts) >= threshold_deg)
    return {"sim_time": sim_time.isoformat(), "iot_cluster": iot_name, "visible_count": count}

@app.get("/api/data/summary", tags=["API/Data"])
def get_data_summary(
    detail: bool = Query(False, description="위성(클라이언트)별 상세 목록 포함"),
    histogram: bool = Query(False, description="위성별 클래스 분포 포함(10 클래스)"),
    limit: int = Query(20, ge=1, le=1000, description="detail=false일 때 나열할 위성 수"),
):
    """
    CIFAR-10 분할/할당 상태 요약
    - 기본: 전역 설정/수량 요약만 반환
    - detail=true: 위성별 샘플 수 포함(기본 최대 limit개)
    - histogram=true: 위성별 클래스 히스토그램(0~9) 포함
    """
    # 데이터 레지스트리/헬퍼 확인
    if "DATA_REGISTRY" not in globals() or "get_training_dataset" not in globals():
        return {"error": "DATA_REGISTRY / get_training_dataset 가 초기화되지 않았습니다."}

    # 구성값/환경
    try:
        cfg = {
            "samples_per_client": SAMPLES_PER_CLIENT,
            "dirichlet_alpha": DIRICHLET_ALPHA,
            "seed": RNG_SEED,
            "with_replacement": WITH_REPLACEMENT,
            "cifar_root": str(CIFAR_ROOT),
        }
    except Exception as e:
        cfg = {"error": f"config fetch failed: {e}"}

    # 위성/할당 현황
    sat_ids = sorted(satellites.keys())
    assignments = getattr(DATA_REGISTRY, "assignments", {})
    num_clients_assigned = len(assignments) if isinstance(assignments, dict) else len(sat_ids)

    result: Dict[str, Any] = {
        "dataset": "CIFAR-10",
        "config": cfg,
        "counts": {
            "num_satellites": len(sat_ids),
            "num_clients_assigned": num_clients_assigned,
        },
        "per_client": [],
        "notes": [
            "detail=true 로 전체 위성 별 샘플 수를 포함할 수 있습니다.",
            "histogram=true 로 (0~9) 클래스 분포를 포함할 수 있습니다.",
            "detail=false 인 경우 limit 만큼만 나열합니다.",
        ],
    }

    # 상세가 아니면 샘플 수 집계를 건너뛰고 끝
    if not detail and not histogram:
        return result

    # 상세 목록 만들기 (detail=false면 limit 만큼만)
    target_ids = sat_ids if detail else sat_ids[:limit]
    total_listed_samples = 0

    for sid in target_ids:
        try:
            ds = get_training_dataset(sid)  # TensorDataset 예상
            n = len(ds) if hasattr(ds, "__len__") else None
            entry: Dict[str, Any] = {"sat_id": sid, "num_samples": n}
            total_listed_samples += (n or 0)

            if histogram and n and hasattr(ds, "tensors") and len(ds.tensors) >= 2:
                labels = ds.tensors[1]
                hist: Dict[int, int] = {}
                try:
                    # torch가 있다면 torch로
                    if _has_torch:
                        import torch
                        uniq, cnt = torch.unique(labels.cpu(), return_counts=True)
                        hist = {int(u.item()): int(c.item()) for u, c in zip(uniq, cnt)}
                    else:
                        # torch 미사용 대비
                        arr = labels.detach().cpu().numpy()
                        c = Counter(arr.tolist())
                        hist = {int(k): int(v) for k, v in c.items()}
                except Exception as e:
                    entry["hist_error"] = f"{e}"
                else:
                    # 0~9 클래스 누락된 항목 0으로 채우기(보기 좋게)
                    for cls in range(10):
                        hist.setdefault(cls, 0)
                    entry["class_hist"] = hist
            result["per_client"].append(entry)
        except Exception as e:
            result["per_client"].append({"sat_id": sid, "error": str(e)})

    # 참고 합계(나열된 항목 기준)
    result["counts"]["total_samples_listed"] = total_listed_samples
    if not detail:
        result["counts"]["listed_clients"] = len(target_ids)

    return result

@app.get("/api/model/global", tags=["API/Model"])
def get_global_model_info(detail: bool = Query(False, description="모든 글로벌 체크포인트 목록/메타데이터 포함")):
    """
    글로벌 모델의 현재 버전/경로를 조회합니다.
    - version: GLOBAL_MODEL_VERSION (초기화 전이면 -1)
    - latest_ckpt: CKPT_DIR/global_v{version}.ckpt 가 존재하면 그 경로를, 없으면 가장 최근 파일을 찾아 반환
    - detail=true 일 때: CKPT_DIR 안의 global_v*.ckpt 파일 리스트와 파일 메타(크기, mtime)를 함께 반환
    """
    # 안전장치: 필요한 전역이 없을 수 있을 때 기본값
    version = globals().get("GLOBAL_MODEL_VERSION", -1)
    ckpt_dir = globals().get("CKPT_DIR", Path(__file__).parent / "ckpt")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # 우선 canonical 경로(global_v{ver}.ckpt)를 시도
    canonical = ckpt_dir / f"global_v{version}.ckpt" if version >= 0 else None
    latest_path = canonical if (canonical and canonical.exists()) else None

    # 없으면 가장 최신 파일을 스캔
    if latest_path is None:
        # _find_latest_global_ckpt()가 있다면 활용
        if "_find_latest_global_ckpt" in globals():
            try:
                best_ver, best_path = _find_latest_global_ckpt()
                if best_ver is not None and best_path is not None and best_path.exists():
                    version = max(version, best_ver)
                    latest_path = best_path
            except Exception:
                pass
        # 그래도 못 찾으면 패턴 매칭으로 스캔
        if latest_path is None:
            candidates = sorted(ckpt_dir.glob("global_v*.ckpt"))
            if candidates:
                latest_path = candidates[-1]

    resp = {
        "version": int(version),
        "latest_ckpt": str(latest_path) if latest_path else None,
        "initialized": bool(version >= 0 and latest_path is not None),
    }

    if detail:
        files = []
        for p in sorted(ckpt_dir.glob("global_v*.ckpt")):
            try:
                stat = p.stat()
                files.append({
                    "path": str(p),
                    "size_bytes": stat.st_size,
                    "mtime_epoch": int(stat.st_mtime),
                    "mtime_iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(stat.st_mtime)),
                })
            except Exception:
                files.append({"path": str(p)})
        resp["files"] = files

    return resp


@app.get("/api/model/global/version", tags=["API/Model"])
def get_global_model_version():
    """
    글로벌 모델 버전 숫자만 간단히 반환합니다.
    """
    version = globals().get("GLOBAL_MODEL_VERSION", -1)
    return {"version": int(version)}

# ==================== 기타 ====================
async def auto_resume_after_delay():
    global sim_paused, auto_resume_delay_sec
    await asyncio.sleep(auto_resume_delay_sec)
    sim_paused = False
    auto_resume_delay_sec = 0
