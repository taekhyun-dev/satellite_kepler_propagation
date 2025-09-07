# simserver/core/cleanup.py
from __future__ import annotations
import os, re, json, shutil, time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Dict, List, Tuple

import torch  # type: ignore

from .paths import CKPT_DIR, LAST_GLOBAL_PTR
from ..core.logging import make_logger

logger = make_logger("simserver.cleanup")

_GLOBAL_ARCH_RE = re.compile(r"^global_v(\d+)_(\d{8}_\d{6})\.ckpt$")
_GLOBAL_LINK_RE = re.compile(r"^global_v(\d+)\.ckpt$")
_SAT_RE = re.compile(
    r"^sat(?P<sid>\d+)_fromg(?P<fromg>-?\d+)_round(?P<rnd>-?\d+)_ep(?P<ep>-?\d+)_\d{8}_\d{6}\.ckpt$"
)

def _now_ts() -> str:
    return __import__("datetime").datetime.now().strftime("%Y%m%d_%H%M%S")

@dataclass
class PrunePolicy:
    # 보존 개수/기간/용량 제한(환경변수로 오버라이드 가능)
    keep_global_arch: int = 300         # timestamp 포함 아카이브 ckpt 보존 개수
    keep_global_link: int = 300         # global_v{ver}.ckpt (타임스탬프 없는 링크) 보존 개수
    keep_sat_per_client: int = 3      # 위성별 sat ckpt 보존 개수
    keep_topk_global_by_val: int = 0    # global_metrics.csv 기준 상위 val acc K개 보호(옵션)FL_PRUNE_ON_SAVE
    max_age_days: Optional[int] = None  # 이 일수보다 오래된 파일 제거(보호 대상 제외)
    max_total_gb: Optional[float] = None# CKPT_DIR 총 용량 상한(초과분부터 오래된 것 제거)
    skip_recent_secs: int = 60          # 최근 n초 이내 생성/수정 파일은 스킵(경합 안전)
    dry_run: bool = False               # True면 실제 삭제 대신 로그만

    # 휴지통으로 이동(fast rollback)
    use_trash: bool = True
    trash_dir: Path = CKPT_DIR / "trash"

def _from_env(policy: PrunePolicy) -> PrunePolicy:
    def _get(name: str, cast, default):
        v = os.getenv(name)
        if v is None: return default
        try: return cast(v)
        except Exception: return default

    return PrunePolicy(
        keep_global_arch = _get("FL_CKPT_KEEP_N_GLOBAL", int, policy.keep_global_arch),
        keep_global_link = _get("FL_CKPT_KEEP_N_GLOBAL_LINK", int, policy.keep_global_link),
        keep_sat_per_client = _get("FL_CKPT_KEEP_N_SAT_PER_CLIENT", int, policy.keep_sat_per_client),
        keep_topk_global_by_val = _get("FL_CKPT_KEEP_TOPK", int, policy.keep_topk_global_by_val),
        max_age_days = _get("FL_CKPT_MAX_AGE_DAYS", int, policy.max_age_days),
        max_total_gb = _get("FL_CKPT_MAX_TOTAL_GB", float, policy.max_total_gb),
        skip_recent_secs = _get("FL_CKPT_SKIP_RECENT_SECS", int, policy.skip_recent_secs),
        dry_run = (_get("FL_CKPT_DRY_RUN", int, 1 if policy.dry_run else 0) == 1),
        use_trash = (_get("FL_CKPT_TRASH", int, 1 if policy.use_trash else 0) == 1),
        trash_dir = Path(os.getenv("FL_CKPT_TRASH_DIR", str(policy.trash_dir))),
    )

def _is_recent(path: Path, skip_recent_secs: int) -> bool:
    try:
        mtime = path.stat().st_mtime
        return (time.time() - mtime) < max(0, skip_recent_secs)
    except Exception:
        return True

def _load_protected_from_pointers() -> set[Path]:
    """LAST_GLOBAL_PTR 와 각 sat*_last.json 이 참조하는 ckpt는 보호."""
    protected: set[Path] = set()
    # 1) 글로벌 포인터
    try:
        if LAST_GLOBAL_PTR.exists():
            with LAST_GLOBAL_PTR.open("r", encoding="utf-8") as f:
                d = json.load(f)
            p = Path(d.get("path", ""))
            if p.exists(): protected.add(p.resolve())
    except Exception as e:
        logger.warning(f"[PRUNE] read LAST_GLOBAL_PTR failed: {e}")

    # 2) 로컬(sat) 포인터들
    try:
        for meta in CKPT_DIR.glob("sat*_last.json"):
            try:
                with meta.open("r", encoding="utf-8") as f:
                    d = json.load(f)
                p = Path(d.get("path", ""))
                if p.exists(): protected.add(p.resolve())
            except Exception:
                continue
    except Exception as e:
        logger.warning(f"[PRUNE] read sat*_last.json failed: {e}")
    return protected

def _load_topk_global_from_metrics(k: int) -> set[Path]:
    """global_metrics.csv에서 val split 기준 상위 K ckpt 보호(옵션)."""
    if k <= 0: return set()
    paths: set[Path] = set()
    csv_path = Path(os.getenv("FL_GLOBAL_METRICS_CSV", str(CKPT_DIR.parent / "global_metrics.csv")))
    if not csv_path.exists(): return set()
    try:
        import csv
        rows = []
        with csv_path.open("r", encoding="utf-8") as f:
            rdr = csv.DictReader(f)
            for r in rdr:
                if (r.get("split") or "").strip() != "val": continue
                try:
                    acc = float(r.get("acc", "nan"))
                except Exception:
                    continue
                path = Path(r.get("ckpt_path",""))
                rows.append((acc, path))
        rows.sort(key=lambda x: x[0], reverse=True)
        for _, p in rows[:k]:
            if p and p.exists(): paths.add(p.resolve())
    except Exception as e:
        logger.warning(f"[PRUNE] read metrics failed: {e}")
    return paths

def _classify_ckpt(p: Path):
    name = p.name
    m1 = _GLOBAL_ARCH_RE.match(name)
    if m1:
        return ("global_arch", int(m1.group(1)), m1.group(2))
    m2 = _GLOBAL_LINK_RE.match(name)
    if m2:
        return ("global_link", int(m2.group(1)), None)
    m3 = _SAT_RE.match(name)
    if m3:
        sid = int(m3.group("sid"))
        rnd = int(m3.group("rnd"))
        ep  = int(m3.group("ep"))
        return ("sat", sid, (rnd, ep))
    return ("other", None, None)

def _size_bytes(p: Path) -> int:
    try: return p.stat().st_size
    except Exception: return 0

def _move_to_trash(p: Path, trash_dir: Path):
    trash_dir.mkdir(parents=True, exist_ok=True)
    dst = trash_dir / f"{_now_ts()}__{p.name}"
    try:
        shutil.move(str(p), str(dst))
        return dst
    except Exception as e:
        logger.warning(f"[PRUNE] move->trash failed for {p}: {e}")
        return None

def _delete(p: Path, *, policy: PrunePolicy):
    if policy.dry_run:
        logger.info(f"[PRUNE] (dry-run) remove {p}")
        return
    if policy.use_trash:
        dst = _move_to_trash(p, policy.trash_dir)
        if dst:
            logger.info(f"[PRUNE] moved to trash: {p} -> {dst}")
        else:
            try:
                p.unlink()
                logger.info(f"[PRUNE] removed {p}")
            except Exception as e:
                logger.warning(f"[PRUNE] unlink failed {p}: {e}")
    else:
        try:
            p.unlink()
            logger.info(f"[PRUNE] removed {p}")
        except Exception as e:
            logger.warning(f"[PRUNE] unlink failed {p}: {e}")

def prune_checkpoints(policy: Optional[PrunePolicy] = None) -> Dict[str, int]:
    """
    CKPT_DIR 의 .ckpt 파일을 정리.
    보호 대상:
      - LAST_GLOBAL_PTR 가 가리키는 ckpt
      - sat*_last.json 이 가리키는 ckpt
      - (옵션) global_metrics.csv 기준 상위 K(val acc)
      - (안전) 최근 skip_recent_secs 이내 수정 파일
    규칙:
      - global_v{ver}.ckpt(링크형): 최신 버전 기준 keep_global_link 개수만 남김
      - global_v{ver}_{ts}.ckpt(아카이브형): 최신부터 keep_global_arch 개수만 남김
      - sat{sid}_...: 위성별 최신 keep_sat_per_client 개수만 남김
      - (옵션) max_age_days, max_total_gb 로 추가 정리
    """
    policy = _from_env(policy or PrunePolicy())
    kept = 0
    removed = 0

    all_ckpt = list(CKPT_DIR.glob("*.ckpt"))
    if not all_ckpt:
        logger.info("[PRUNE] no ckpt files found")
        return {"kept": 0, "removed": 0}

    # 보호 집합 수집
    protected = _load_protected_from_pointers()
    protected |= _load_topk_global_from_metrics(policy.keep_topk_global_by_val)

    # 최근 파일 보호
    recent_protect: set[Path] = set(p for p in all_ckpt if _is_recent(p, policy.skip_recent_secs))

    # 분류
    groups: Dict[str, List[Path]] = {"global_link": [], "global_arch": [], "sat": [], "other": []}
    for p in all_ckpt:
        kind, _, _ = _classify_ckpt(p)
        groups.setdefault(kind, []).append(p)

    # 1) global_link: 버전 내림차순으로 keep
    def _ver(p: Path) -> int:
        m = _GLOBAL_LINK_RE.match(p.name)
        return int(m.group(1)) if m else -1
    keep_gl = sorted(groups["global_link"], key=_ver, reverse=True)[:max(0, policy.keep_global_link)]
    kill_gl = [p for p in groups["global_link"] if p not in keep_gl]

    # 2) global_arch: 타임스탬프 포함 아카이브 최신 keep
    def _arch_key(p: Path) -> Tuple[int, str]:
        m = _GLOBAL_ARCH_RE.match(p.name)
        if not m: return (-1, "")
        return (int(m.group(1)), m.group(2))
    keep_ga = sorted(groups["global_arch"], key=_arch_key, reverse=True)[:max(0, policy.keep_global_arch)]
    kill_ga = [p for p in groups["global_arch"] if p not in keep_ga]

    # 3) sat: 위성별 최신 keep_sat_per_client 유지
    sat_by_id: Dict[int, List[Path]] = {}
    for p in groups["sat"]:
        m = _SAT_RE.match(p.name)
        sid = int(m.group("sid")) if m else -1
        sat_by_id.setdefault(sid, []).append(p)
    kill_sat: List[Path] = []
    for sid, lst in sat_by_id.items():
        lst_sorted = sorted(lst, key=lambda x: x.stat().st_mtime, reverse=True)
        keep = lst_sorted[:max(0, policy.keep_sat_per_client)]
        kill_sat.extend([p for p in lst_sorted if p not in keep])

    # 4) other: 일단 보존(명시 삭제 X)
    kill_other: List[Path] = []

    # max_age_days 규칙 적용(보호/최근 제외)
    aged_out: List[Path] = []
    if policy.max_age_days is not None:
        cutoff = time.time() - policy.max_age_days * 86400
        for p in all_ckpt:
            if p.resolve() in protected or p in recent_protect:
                continue
            try:
                if p.stat().st_mtime < cutoff:
                    aged_out.append(p)
            except Exception:
                continue

    # 1차 삭제 목록
    to_remove = set(kill_gl + kill_ga + kill_sat + kill_other + aged_out)

    # 보호 대상 제외
    to_remove = {p for p in to_remove if p.resolve() not in protected and p not in recent_protect}

    # 실행
    for p in sorted(to_remove, key=lambda x: x.stat().st_mtime if x.exists() else 0):
        _delete(p, policy=policy)
        removed += 1

    # 용량 상한 검사(선택)
    if policy.max_total_gb is not None:
        usable = sorted([p for p in CKPT_DIR.glob("*.ckpt") if p.exists()],
                        key=lambda x: x.stat().st_mtime)
        total = sum(_size_bytes(p) for p in usable)
        limit = int(policy.max_total_gb * (1024**3))
        idx = 0
        while total > limit and idx < len(usable):
            p = usable[idx]
            if (p.resolve() in protected) or (p in recent_protect):
                idx += 1; continue
            _delete(p, policy=policy)
            total -= _size_bytes(p)
            removed += 1
            idx += 1

    kept = len(list(CKPT_DIR.glob("*.ckpt")))
    logger.info(f"[PRUNE] done kept={kept} removed={removed} (dry_run={policy.dry_run})")
    return {"kept": kept, "removed": removed}
