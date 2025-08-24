# simserver/core/logging.py
import logging

def make_logger(name: str = "simserver", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if "." not in name:
        # 최상위 패키지 로거: 핸들러 1개만, root로 전파 차단
        if not logger.handlers:
            h = logging.StreamHandler()
            h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
            logger.addHandler(h)
        logger.propagate = False
    else:
        # 하위 로거: 핸들러 제거(혹시 이전에 붙었으면), 부모로 전파 허용
        if logger.handlers:
            logger.handlers.clear()
        logger.propagate = True

    return logger

def exc_repr(e: Exception) -> str:
    try:
        return f"{type(e).__name__}: {e}"
    except Exception:
        return f"{type(e).__name__}"
