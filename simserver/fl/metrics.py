# simserver/fl/metrics.py
from pathlib import Path
from typing import List, Any
import threading
from ..core.paths import GLOBAL_METRICS_CSV, GLOBAL_METRICS_XLSX, GLOBAL_METRICS_HEADER
from ..core.paths import LOCAL_METRICS_DIR
from ..core.paths import is_zip_file
from ..core.logging import make_logger

logger = make_logger("simserver.metrics")
_METRICS_LOCK = threading.Lock()

def _append_csv_row(path: Path, header: List[str], row: List[Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with _METRICS_LOCK:
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
    import os
    if os.getenv("FL_DISABLE_XLSX", "0") in ("1","true","True"):
        return
    try:
        import pandas as pd
    except Exception:
        return
    xlsx_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df = None
        if xlsx_path.exists():
            if not is_zip_file(xlsx_path):
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
        logger.warning(f"[METRICS] Excel write failed: {e}")

def log_local_metrics(sat_id: int, round_idx: int, epoch: int, loss: float, acc: float, n: int, ckpt_path: str):
    ts = __import__("datetime").datetime.now().isoformat()
    header = ["timestamp", "sat_id", "round", "epoch", "num_samples", "loss", "acc", "ckpt_path"]
    row = [ts, sat_id, round_idx, epoch, n, f"{loss:.6f}", f"{acc:.2f}", ckpt_path]
    csv_path = LOCAL_METRICS_DIR / f"sat{sat_id}.csv"
    _append_csv_row(csv_path, header, row)
    _append_excel_row(csv_path.with_suffix(".xlsx"), header, row, sheet="local")

def log_global_metrics(version: int, split: str, loss: float, acc: float, n: int,
                       ckpt_path: str, *, f1_macro=None, madds_M=None, flops_M=None, latency_ms=None):
    ts = __import__("datetime").datetime.now().isoformat()
    row = [ts, version, split, n, f"{loss:.6f}", f"{acc:.2f}",
           (f"{f1_macro:.2f}" if f1_macro is not None else ""),
           (f"{madds_M:.2f}"  if madds_M is not None  else ""),
           (f"{flops_M:.2f}"  if flops_M is not None  else ""),
           (f"{latency_ms:.3f}" if latency_ms is not None else ""),
           ckpt_path]
    _append_csv_row(GLOBAL_METRICS_CSV, GLOBAL_METRICS_HEADER, row)
    _append_excel_row(GLOBAL_METRICS_XLSX, GLOBAL_METRICS_HEADER, row, sheet="global")
