# simserver/dataio/registry.py
# 원래의 data 모듈과 동일 함수명을 얇게 감싸, 의존 방향을 단순화
from typing import Any
from pathlib import Path

try:
    from ..data import (
        CIFAR_ROOT, ASSIGNMENTS_DIR, SAMPLES_PER_CLIENT, DIRICHLET_ALPHA, RNG_SEED, WITH_REPLACEMENT,
        DATA_REGISTRY, get_training_dataset,
    )
except Exception as e:
    raise ImportError("data module not found or incomplete") from e
