from pathlib import Path
import os

# 폴더 경로
DATA_DIR = Path(__file__).resolve().parent
CIFAR_ROOT = DATA_DIR / "cifar_data"        # torchvision이 여기로 받음
ASSIGNMENTS_DIR = DATA_DIR / "assignments"
ASSIGNMENTS_DIR.mkdir(parents=True, exist_ok=True)
CIFAR_ROOT.mkdir(parents=True, exist_ok=True)

# 환경변수 기반 기본값
SAMPLES_PER_CLIENT = int(os.getenv("FL_SAMPLES_PER_CLIENT", "2000"))
DIRICHLET_ALPHA    = float(os.getenv("FL_DIRICHLET_ALPHA", "0.3"))
RNG_SEED           = int(os.getenv("FL_DATA_SEED", "42"))
WITH_REPLACEMENT   = os.getenv("FL_WITH_REPLACEMENT", "1") != "0"  # "0"이면 False
