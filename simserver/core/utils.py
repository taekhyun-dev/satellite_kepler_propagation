# simserver/core/utils.py
import os

def get_env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return int(default)

def get_env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return float(default)

def is_zip_file(path) -> bool:
    try:
        import zipfile
        return zipfile.is_zipfile(path)
    except Exception:
        return False
