# simserver/core/logging.py
import logging, os, sys, json, time
from logging.handlers import TimedRotatingFileHandler, RotatingFileHandler
from pathlib import Path
from .paths import LOG_DIR

_LOGGING_CONFIGURED = False


def _to_bool(s: str, default=False):
    if s is None: return default
    return s.strip().lower() in {"1","true","yes","on"}

class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(record.created)),
            "level": record.levelname,
            "name": record.name,
            "msg": record.getMessage(),
            "process": record.process,
            "thread": record.thread,
            "module": record.module,
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)

def _text_formatter():
    return logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p

def setup_logging(
    log_dir: Path | None = None,
    filename: str | None = None,
    level: str | None = None,
):
    """
    루트 로거에 콘솔 + 파일 핸들러를 한 번만 설정합니다.
    - 환경변수:
      FL_LOG_DIR: 로그 디렉토리 (기본: paths.LOG_DIR)
      FL_LOG_FILE: 파일명 (기본: simserver.log)
      FL_LOG_LEVEL: DEBUG/INFO/WARNING/ERROR (기본: INFO)
      FL_LOG_JSON: 1이면 JSON 라인 포맷 (기본: 0)
      FL_LOG_ROTATE: time/size/none (기본: time)
      FL_LOG_WHEN: TimedRotating when (기본: midnight)
      FL_LOG_BACKUPS: 보관 파일 개수 (기본: 10)
      FL_LOG_MAX_BYTES: size 로테이션 시 최대 바이트 (기본: 50MB)
    """
    global _LOGGING_CONFIGURED
    if _LOGGING_CONFIGURED:
        return

    log_dir = Path(os.getenv("FL_LOG_DIR", str(log_dir or LOG_DIR)))
    filename = os.getenv("FL_LOG_FILE", filename or "simserver.log")
    level = (os.getenv("FL_LOG_LEVEL", level or "INFO")).upper()

    use_json = _to_bool(os.getenv("FL_LOG_JSON", "0"))
    rotate_mode = os.getenv("FL_LOG_ROTATE", "time").lower()  # time/size/none
    when = os.getenv("FL_LOG_WHEN", "midnight")
    backups = int(os.getenv("FL_LOG_BACKUPS", "10"))
    max_bytes = int(os.getenv("FL_LOG_MAX_BYTES", str(50 * 1024 * 1024)))  # 50MB

    _ensure_dir(log_dir)
    log_path = log_dir / filename

    root = logging.getLogger()
    root.setLevel(getattr(logging, level, logging.INFO))

    # 포맷터
    formatter = _JsonFormatter() if use_json else _text_formatter()

    # 콘솔
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(getattr(logging, level, logging.INFO))
    sh.setFormatter(formatter)
    root.addHandler(sh)

    # 파일 (로테이션)
    if rotate_mode == "none":
        fh = logging.FileHandler(str(log_path), encoding="utf-8")
    elif rotate_mode == "size":
        fh = RotatingFileHandler(str(log_path), maxBytes=max_bytes, backupCount=backups, encoding="utf-8")
    else:
        fh = TimedRotatingFileHandler(str(log_path), when=when, backupCount=backups, encoding="utf-8", utc=False)
    fh.setLevel(getattr(logging, level, logging.INFO))
    fh.setFormatter(formatter)
    root.addHandler(fh)

    # latest.log 심볼릭 링크(편의)
    try:
        latest = log_dir / "latest.log"
        if latest.exists() or latest.is_symlink():
            latest.unlink()
        latest.symlink_to(log_path.name)
    except Exception:
        pass

    # 파이썬 warnings도 로깅으로
    logging.captureWarnings(True)

    _LOGGING_CONFIGURED = True

def make_logger(name: str) -> logging.Logger:
    """
    모듈별 로거 생성. setup_logging()에서 붙인 루트 핸들러를 사용합니다.
    """
    return logging.getLogger(name)

def exc_repr(e: Exception) -> str:
    try:
        return f"{type(e).__name__}: {e}"
    except Exception:
        return f"{type(e).__name__}"
