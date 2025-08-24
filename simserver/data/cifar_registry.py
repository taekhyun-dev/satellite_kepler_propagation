from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path

import os
import torch
from torch.utils.data import Dataset, Subset

try:
    import torchvision
    from torchvision.datasets import CIFAR10
    from torchvision import transforms
except Exception:
    torchvision, CIFAR10 = None, None, None

from .config import CIFAR_ROOT, SAMPLES_PER_CLIENT, DIRICHLET_ALPHA, RNG_SEED, WITH_REPLACEMENT


class CIFARSubsetDataset(Dataset):
    """CIFAR-10 원본(uint8)을 공유하고, 인덱스 배열만 복제 → 메모리 절약."""
    MEAN = torch.tensor([0.4915, 0.4823, 0.4468], dtype=torch.float32)
    STD  = torch.tensor([0.2470, 0.2435, 0.2616], dtype=torch.float32)

    def __init__(self, base_x_uint8: np.ndarray, base_y: np.ndarray, indices: np.ndarray):
        self.x = base_x_uint8           # (N, 32, 32, 3) uint8
        self.y = base_y                 # (N,)
        self.idx = indices.astype(np.int64)

    def __len__(self):
        return int(self.idx.shape[0])

    def __getitem__(self, i: int):
        j = int(self.idx[i])
        # HWC uint8 -> CHW float32 (0~1) -> 채널 정규화
        x = torch.from_numpy(self.x[j]).permute(2, 0, 1).to(dtype=torch.float32) / 255.0
        x = (x - self.MEAN[:, None, None]) / self.STD[:, None, None]
        y = int(self.y[j])
        return x, y


class CIFARDataRegistry:
    """
    - CIFAR-10을 한 번만 로드(원본 uint8 유지)
    - 위성(sat_id)별로 샘플 인덱스만 보유(중복 샘플링으로 '데이터셋 크기' 임의 확대 가능)
    """
    def __init__(self):
        self.base_x: Optional[np.ndarray] = None  # (N,32,32,3) uint8
        self.base_y: Optional[np.ndarray] = None  # (N,)
        self.class_idx: Dict[int, np.ndarray] = {}
        self.client_idx: Dict[int, np.ndarray] = {}
        self.rng = np.random.default_rng(RNG_SEED)

        # ↓ 평가용 캐시/스플릿
        self._eval_split_ready = False
        self._val_idx: Optional[np.ndarray] = None
        self._test_idx: Optional[np.ndarray] = None

    def load_base(self, root: Path = CIFAR_ROOT, seed: int = RNG_SEED, download: bool = True):
        if torchvision is None or CIFAR10 is None:
            raise RuntimeError("torchvision이 필요합니다. `pip install torchvision` 후 재시도하세요.")
        self.rng = np.random.default_rng(seed)
        ds = CIFAR10(root=str(root), train=True, download=download)
        self.base_x = np.array(ds.data, dtype=np.uint8)      # (50000,32,32,3)
        self.base_y = np.array(ds.targets, dtype=np.int64)   # (50000,)
        self.class_idx = {c: np.where(self.base_y == c)[0] for c in range(10)}

    def _sample_dirichlet_counts(self, total: int, alpha: float, num_classes=10) -> np.ndarray:
        p = self.rng.dirichlet([alpha] * num_classes)
        counts = np.floor(p * total).astype(int)
        diff = total - int(counts.sum())
        if diff > 0:
            order = np.argsort(-p)
            counts[order[:diff]] += 1
        return counts

    def assign_clients(
        self,
        sat_ids: List[int],
        samples_per_client: int = SAMPLES_PER_CLIENT,
        alpha: float = DIRICHLET_ALPHA,
        seed: int = RNG_SEED,
        with_replacement: bool = WITH_REPLACEMENT,
    ):
        assert self.base_x is not None, "먼저 load_base()를 호출하세요."
        self.rng = np.random.default_rng(seed)
        for sid in sat_ids:
            counts = self._sample_dirichlet_counts(samples_per_client, alpha, num_classes=10)
            per_class_indices = []
            for c in range(10):
                pool = self.class_idx[c]
                k = int(counts[c])
                if with_replacement:
                    choice = self.rng.choice(pool, size=k, replace=True)
                else:
                    if k <= len(pool):
                        choice = self.rng.choice(pool, size=k, replace=False)
                    else:
                        a = self.rng.choice(pool, size=len(pool), replace=False)
                        b = self.rng.choice(pool, size=k - len(pool), replace=True)
                        choice = np.concatenate([a, b], axis=0)
                per_class_indices.append(choice)
            idx = np.concatenate(per_class_indices, axis=0)
            self.rng.shuffle(idx)
            self.client_idx[sid] = idx

    def get_dataset(self, sat_id: int) -> Dataset:
        if sat_id not in self.client_idx:
            # 동적 기본 할당(필요 시)
            self.assign_clients([sat_id], samples_per_client=SAMPLES_PER_CLIENT, alpha=DIRICHLET_ALPHA)
        return CIFARSubsetDataset(self.base_x, self.base_y, self.client_idx[sat_id])

    # ---------- 선택: 인덱스 캐시 저장/불러오기 ----------
    def save_assignments(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        # np.savez로 {sid: indices} 저장
        np.savez_compressed(str(path), **{f"sid_{k}": v for k, v in self.client_idx.items()})

    def load_assignments(self, path: Path) -> bool:
        if not path.exists():
            return False
        data = np.load(str(path))
        loaded = {}
        for key in data.files:
            if not key.startswith("sid_"):
                continue
            sid = int(key.split("_", 1)[1])
            loaded[sid] = data[key]
        if loaded:
            self.client_idx = loaded
            return True
        return False

    def _eval_transform(self):
        """Resize + Normalize(transform) 생성"""
        if transforms is None:
            return None
        img_size = int(os.getenv("FL_IMG_SIZE", "32"))
        # 환경변수 우선, 없으면 CIFARSubsetDataset의 통계 사용
        mean_env = os.getenv("FL_NORM_MEAN")
        std_env  = os.getenv("FL_NORM_STD")
        if mean_env and std_env:
            mean = tuple(map(float, mean_env.split(",")))
            std  = tuple(map(float, std_env.split(",")))
        else:
            mean = tuple(float(x) for x in CIFARSubsetDataset.MEAN.tolist())
            std  = tuple(float(x) for x in CIFARSubsetDataset.STD.tolist())
        return transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    def _ensure_eval_splits(self):
        """CIFAR-10 test(10k)를 고정 시드로 val/test로 나눠 인덱스 캐시."""
        if self._eval_split_ready or CIFAR10 is None:
            return
        ds = CIFAR10(root=str(CIFAR_ROOT), train=False, download=True)
        N = len(ds)  # 10000
        val_frac = float(os.getenv("FL_VAL_FRACTION", "0.5"))
        n_val = max(1, min(N - 1, int(N * val_frac)))
        rng = np.random.default_rng(RNG_SEED)
        idx = np.arange(N)
        rng.shuffle(idx)
        self._val_idx = idx[:n_val]
        self._test_idx = idx[n_val:]
        self._eval_split_ready = True

    def get_validation_dataset(self) -> Optional[Dataset]:
        """검증용 데이터셋 반환(없으면 None). 기본은 CIFAR-10 test의 절반 서브셋."""
        if torchvision is None or CIFAR10 is None:
            return None
        self._ensure_eval_splits()
        tfm = self._eval_transform()
        ds_full = CIFAR10(root=str(CIFAR_ROOT), train=False, download=True, transform=tfm)
        if self._val_idx is None:
            # 스플릿 못 만들면 전체를 반환(최소 동작 보장)
            return ds_full
        return Subset(ds_full, self._val_idx.tolist())

    def get_test_dataset(self) -> Optional[Dataset]:
        """테스트용 데이터셋 반환(없으면 None). 기본은 CIFAR-10 test의 나머지 절반."""
        if torchvision is None or CIFAR10 is None:
            return None
        self._ensure_eval_splits()
        tfm = self._eval_transform()
        ds_full = CIFAR10(root=str(CIFAR_ROOT), train=False, download=True, transform=tfm)
        if self._test_idx is None:
            # 스플릿 못 만들면 전체를 반환(최소 동작 보장)
            return ds_full
        return Subset(ds_full, self._test_idx.tolist())


# ---- 전역 싱글톤 & 간편 훅 ----
DATA_REGISTRY = CIFARDataRegistry()

def get_training_dataset(sat_id: int) -> Dataset:
    return DATA_REGISTRY.get_dataset(sat_id)

def get_validation_dataset() -> Optional[Dataset]:
    return DATA_REGISTRY.get_validation_dataset()

def get_test_dataset() -> Optional[Dataset]:
    return DATA_REGISTRY.get_test_dataset()