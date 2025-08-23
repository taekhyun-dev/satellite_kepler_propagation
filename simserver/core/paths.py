# simserver/core/paths.py
from pathlib import Path
from .utils import is_zip_file

BASE_DIR = Path(__file__).resolve().parents[1]  # project root/simserver
CKPT_DIR = BASE_DIR / "ckpt"
METRICS_DIR = BASE_DIR / "metrics"
GLOBAL_METRICS_DIR = METRICS_DIR / "global"
LOCAL_METRICS_DIR = METRICS_DIR / "local"
LAST_GLOBAL_PTR = CKPT_DIR / "last_global.json"
GLOBAL_METRICS_CSV = GLOBAL_METRICS_DIR / "global_metrics.csv"
GLOBAL_METRICS_XLSX = GLOBAL_METRICS_DIR / "global_metrics.xlsx"
GLOBAL_METRICS_HEADER = [
    "timestamp","version","split","num_samples","loss","acc","f1_macro",
    "madds_M","flops_M","latency_ms","ckpt_path"
]

for p in [CKPT_DIR, GLOBAL_METRICS_DIR, LOCAL_METRICS_DIR, METRICS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

__all__ = [
    "BASE_DIR","CKPT_DIR","METRICS_DIR","GLOBAL_METRICS_DIR","LOCAL_METRICS_DIR",
    "LAST_GLOBAL_PTR","GLOBAL_METRICS_CSV","GLOBAL_METRICS_XLSX","GLOBAL_METRICS_HEADER","is_zip_file"
]
