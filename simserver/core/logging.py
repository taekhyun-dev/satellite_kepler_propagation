# simserver/core/logging.py
import logging

def make_logger(name="simserver", level=logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        logger.addHandler(h)
    return logger

def exc_repr(e: Exception) -> str:
    try:
        return f"{type(e).__name__}: {e}"
    except Exception:
        return f"{type(e).__name__}"
