# simserver/core/gpu.py
import os
from typing import Optional, List

def _count_visible_gpus_from_env() -> Optional[int]:
    cvd = os.getenv("CUDA_VISIBLE_DEVICES")
    if cvd is None: return None
    cvd = cvd.strip()
    if cvd == "": return 0
    return len([x for x in cvd.split(",") if x.strip() != ""])

def compute_num_gpus(env_override: Optional[int] = None) -> int:
    if env_override is not None:
        vis = _count_visible_gpus_from_env()
        wanted = max(0, int(env_override))
        return min(wanted, vis) if vis is not None else wanted

    vis = _count_visible_gpus_from_env()
    if vis is not None:
        try:
            import torch
            if not torch.cuda.is_available():
                return 0
        except Exception:
            return 0
        return vis

    try:
        import torch
        return torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception:
        return 0

def build_worker_gpu_list(num_gpus: int, sessions_per_gpu: int) -> List[Optional[int]]:
    if num_gpus > 0:
        return [gid for gid in range(num_gpus) for _ in range(sessions_per_gpu)]
    return [None] * max(1, sessions_per_gpu)
