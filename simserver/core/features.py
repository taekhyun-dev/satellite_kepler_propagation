# simserver/core/features.py
from dataclasses import dataclass
import importlib.util

def _has(mod: str) -> bool:
    return importlib.util.find_spec(mod) is not None

@dataclass(frozen=True)
class Features:
    has_torch: bool
    has_cuda: bool
    has_pandas: bool
    has_thop: bool
    has_fvcore: bool

def detect_features() -> Features:
    ht = _has("torch")
    hc = False
    if ht:
        try:
            import torch
            hc = torch.cuda.is_available()
        except Exception:
            hc = False
    return Features(
        has_torch=ht,
        has_cuda=hc,
        has_pandas=_has("pandas"),
        has_thop=_has("thop"),
        has_fvcore=_has("fvcore"),
    )
