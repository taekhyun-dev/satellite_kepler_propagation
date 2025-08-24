# simserver/web/routes_api.py
import time
from fastapi import APIRouter, Query, Request
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from ..sim.state import AppState
from ..sim.skyfiled import to_ts, elevation_deg, get_current_time_utc
from ..dataio.registry import (
    CIFAR_ROOT, SAMPLES_PER_CLIENT, DIRICHLET_ALPHA, RNG_SEED, WITH_REPLACEMENT,
    DATA_REGISTRY, get_training_dataset,
)
from typing import List
from pathlib import Path
from collections import Counter
from ..fl.aggregate import find_latest_global_ckpt

router = APIRouter()

def _ctx(request) -> AppState:
    return request.app.state.ctx  # set in app.py

@router.post("/api/reset_time", tags=["API"])
def reset_sim_time(request: Request):
    ctx = _ctx(request)
    ctx.sim_time = datetime.fromisoformat(ctx.cfg.start_time_iso)
    return {"status": "reset", "sim_time": ctx.sim_time.isoformat()}

@router.get("/api/sim_time", tags=["API"])
def get_sim_time_api(request: Request):
    ctx = _ctx(request)
    return {"sim_time": ctx.sim_time.isoformat()}

@router.put("/api/sim_time", tags=["API"])
def set_sim_time(request: Request, year: int, month: int, day: int, hour: int = 0, minute: int = 0, second: int = 0):
    ctx = _ctx(request)
    try:
        ctx.sim_time = datetime(year, month, day, hour, minute, second)
        return {"status": "updated", "sim_time": ctx.sim_time.isoformat()}
    except Exception as e:
        return {"error": str(e)}

@router.put("/api/set_step", tags=["API"])
def set_step(request: Request, delta_sec: float = Query(...), interval_sec: float = Query(...)):
    ctx = _ctx(request)
    if delta_sec <= 0 or interval_sec < 0:
        return {"error": "delta_sec>0, interval_sec>=0"}
    ctx.sim_delta_sec = float(delta_sec)
    ctx.real_interval_sec = float(interval_sec)
    return {"sim_delta_sec": ctx.sim_delta_sec, "real_interval_sec": ctx.real_interval_sec}

@router.get("/api/trajectory", tags=["API"])
def get_trajectory(request: Request, sat_id: int = Query(...)):
    ctx = _ctx(request)
    if sat_id not in ctx.satellites:
        return {"error": f"sat_id {sat_id} not found"}
    satellite = ctx.satellites[sat_id]
    t0 = ctx.sim_time
    prev_lon, segment, segments = None, [], []
    for offset_sec in range(0, 7200, 30):
        future = t0 + timedelta(seconds=offset_sec)
        subpoint = satellite.at(to_ts(future)).subpoint()
        lat, lon = subpoint.latitude.degrees, subpoint.longitude.degrees
        if prev_lon is not None and abs(lon - prev_lon) > 180:
            segments.append(segment); segment = []
        segment.append({"lat": lat, "lon": lon}); prev_lon = lon
    if segment: segments.append(segment)
    return {"sat_id": sat_id, "segments": segments}

@router.get("/api/position", tags=["API"])
def get_position(request: Request, sat_id: int = Query(...)):
    ctx = _ctx(request)
    if sat_id not in ctx.current_sat_positions:
        return {"error": f"Position for SAT{sat_id} not available"}
    return ctx.current_sat_positions[sat_id]

@router.get("/api/comm_targets", tags=["API"])
def get_comm_targets(request: Request, sat_id: int = Query(...)):
    ctx = _ctx(request)
    if sat_id not in ctx.satellites:
        return {"error": f"sat_id {sat_id} not found"}
    t_ts = to_ts(ctx.sim_time)
    sat = ctx.satellites[sat_id]
    visible_ground = [name for name, gs in ctx.observer_locations.items() if elevation_deg(sat, gs, t_ts) >= ctx.threshold_deg]
    visible_iot = [name for name, cluster in ctx.iot_clusters.items() if elevation_deg(sat, cluster, t_ts) >= ctx.threshold_deg]
    return {"sim_time": ctx.sim_time.isoformat(), "sat_id": sat_id,
            "visible_ground_stations": visible_ground, "visible_iot_clusters": visible_iot}

@router.get("/api/gs/visibility", tags=["API/GS"])
def get_gs_visibility(request: Request):
    ctx = _ctx(request)
    result = {}
    t_ts = to_ts(ctx.sim_time)
    for name, gs in ctx.observer_locations.items():
        visible_sats = []
        for sid, sat in ctx.satellites.items():
            alt_deg = elevation_deg(sat, gs, t_ts)
            if alt_deg >= ctx.threshold_deg:
                visible_sats.append({"sat_id": sid, "elevation": alt_deg})
        result[name] = visible_sats
    return {"sim_time": ctx.sim_time.isoformat(), "data": result}

@router.get("/api/gs/visibility_schedule", tags=["API/GS"])
def get_visibility_schedule(request: Request, sat_id: int = Query(...)):
    """
    특정 위성의 관측 가능 시간대를 반환하는 API
    """
    ctx = _ctx(request)
    if sat_id not in ctx.satellites:
        return {"error": f"sat_id {sat_id} not found"}
    satellite = ctx.satellites[sat_id]
    results = {}

    for name, gs in ctx.observer_locations.items():
        visible_periods: List[tuple[str, str]] = []
        visible = False
        start: Optional[datetime] = None
        for offset in range(0, 7200, 30):
            future = ctx.sim_time + timedelta(seconds=offset)
            t_ts = to_ts(future)
            alt_deg = elevation_deg(satellite, gs, t_ts)
            if alt_deg >= ctx.threshold_deg:
                if not visible:
                    start = future
                    visible = True
            else:
                if visible:
                    visible_periods.append((start.isoformat(), future.isoformat()))
                    visible = False
        if visible and start:
            visible_periods.append((start.isoformat(), future.isoformat()))
        results[name] = visible_periods

    return {"sim_time": ctx.sim_time.isoformat(), "sat_id": sat_id, "schedule": results}

@router.get("/api/gs/next_comm", tags=["API/GS"])
def get_next_comm_for_sat(request: Request, sat_id: int = Query(..., description="위성 ID")):
    """
    주어진 sat_id 위성이 다음에 통신 가능한 지상국과 시간(시뮬레이션 시간)을 반환하는 API
    """
    ctx = _ctx(request)
    if sat_id not in ctx.satellites:
        return {"error": f"sat_id {sat_id} not found"}
    sat = ctx.satellites[sat_id]
    t0 = ctx.sim_time
    horizon = 86400
    step = max(1, int(ctx.sim_delta_sec))

    for offset in range(step, horizon + step, step):
        t_future = t0 + timedelta(seconds=offset)
        if any(elevation_deg(sat, gs, to_ts(t_future)) >= ctx.threshold_deg for gs in ctx.observer_locations.values()):
            # 가장 먼저 통신되는 지상국 이름도 함께 찾기
            for name, gs in ctx.observer_locations.items():
                if elevation_deg(sat, gs, to_ts(t_future)) >= ctx.threshold_deg:
                    return {"sat_id": sat_id, "ground_station": name, "next_comm_time": t_future.isoformat()}
    return {"sat_id": sat_id, "next_comm_time": None, "message": "다음 24시간 내 통신 창 없음"}

@router.get("/api/gs/next_comm_all", tags=["API/GS"])
def get_next_comm_all(request: Request):
    """
    모든 위성에 대해 다음 통신 가능 시간(지상국, 시뮬레이션 시간)을 반환하는 API
    """
    ctx = _ctx(request)
    t0 = ctx.sim_time
    horizon = 86400
    step = max(1, int(ctx.sim_delta_sec))
    result = {}

    for sat_id, sat in ctx.satellites.items():
        next_time = None
        next_gs = None
        for offset in range(step, horizon + step, step):
            t_future = t0 + timedelta(seconds=offset)
            for name, gs in ctx.observer_locations.items():
                if elevation_deg(sat, gs, to_ts(t_future)) >= ctx.threshold_deg:
                    next_time = t_future
                    next_gs = name
                    break
            if next_time:
                break
        result[sat_id] = {
            "ground_station": next_gs,
            "next_comm_time": next_time.isoformat() if next_time else None
        }
    return {"sim_time": ctx.sim_time.isoformat(), "next_comm": result}

@router.put("/api/observer", tags=["API/Observer"])
def set_observer(request: Request, name: str = Query(...)):
    """
    지상국 관측 위치를 설정하는 API
    """
    ctx = _ctx(request)

    global observer, current_observer_name
    if name not in ctx.observer_locations:
        return {"error": f"observer '{name}' is not supported"}
    observer = ctx.observer_locations[name]
    current_observer_name = name
    return {"status": "ok", "observer": name}


@router.get("/api/observer/check_comm", tags=["API/Observer"])
def get_observer_check_comm(request: Request, sat_id: int = Query(...)):
    """
    위성 통신 가능 여부를 확인하는 API
    """
    ctx = _ctx(request)

    available = bool(ctx.sat_comm_status.get(sat_id, False))
    return {"sat_id": sat_id, "observer": current_observer_name, "sim_time": ctx.sim_time.isoformat(), "available": available}


@router.get("/api/observer/all_visible", tags=["API/Observer"])
def get_all_visible(request: Request):
    """
    현재 시뮬레이션 시간에 지상국(observer)에서 관측 가능한 모든 위성의 ID를 반환하는 API
    """
    ctx = _ctx(request)

    visible_sats = [sat_id for sat_id, available in ctx.sat_comm_status.items() if bool(available)]
    return {"observer": current_observer_name, "sim_time": ctx.sim_time.isoformat(), "visible_sat_ids": visible_sats}


@router.get("/api/observer/visible_count", tags=["API/Observer"])
def get_visible_count(request: Request):
    """
    현재 시뮬레이션 시간에 지상국(observer)에서 관측 가능한 위성의 개수를 반환하는 API
    """
    ctx = _ctx(request)

    count = sum(1 for available in ctx.sat_comm_status.values() if bool(available))
    return {"observer": current_observer_name, "sim_time": ctx.sim_time.isoformat(), "visible_count": count}


@router.get("/api/observer/elevation", tags=["API/Observer"])
def get_elevation(request: Request, sat_id: int = Query(...)):
    """
    특정 위성의 현재 시뮬레이션 시간에 대한 지상국과의 고도를 반환하는 API
    """
    ctx = _ctx(request)

    satellite = ctx.satellites.get(sat_id)
    if satellite is None:
        return {"error": f"sat_id {sat_id} not found"}
    alt_deg = elevation_deg(satellite, observer, get_current_time_utc(ctx.sim_time))
    return {"sat_id": sat_id, "observer": current_observer_name, "sim_time": ctx.sim_time.isoformat(), "elevation_deg": alt_deg}


@router.get("/api/observer/next_comm", tags=["API/Observer"])
def get_next_comm_with_observer(request: Request, sat_id: int = Query(..., description="위성 ID")):
    """
    주어진 sat_id 위성이 현재 observer와 다음에 통신 가능한 시간을 반환하는 API
    """
    ctx = _ctx(request)

    if sat_id not in ctx.satellites:
        return {"error": f"sat_id {sat_id} not found"}
    sat = ctx.satellites[sat_id]
    t0 = ctx.sim_time
    horizon = 86400
    step = max(1, int(ctx.sim_delta_sec))

    for offset in range(step, horizon + step, step):
        t_future = t0 + timedelta(seconds=offset)
        if elevation_deg(sat, observer, to_ts(t_future)) >= ctx.threshold_deg:
            return {"sat_id": sat_id, "observer": current_observer_name, "next_comm_time": t_future.isoformat()}
    return {"sat_id": sat_id, "observer": current_observer_name, "next_comm_time": None, "message": "다음 24시간 내 통신 창 없음"}


@router.get("/api/observer/next_comm_all", tags=["API/Observer"])
def get_next_comm_all_with_observer(request: Request):
    """
    모든 위성에 대해 현재 observer 와의 다음 통신 가능한 시간을 반환하는 API
    """
    ctx = _ctx(request)
    
    t0 = ctx.sim_time
    horizon = 86400
    step = max(1, int(ctx.sim_delta_sec))
    result = {}

    for sat_id, sat in ctx.satellites.items():
        next_time = None
        for offset in range(step, horizon + step, step):
            t_future = t0 + timedelta(seconds=offset)
            if elevation_deg(sat, observer, to_ts(t_future)) >= ctx.threshold_deg:
                next_time = t_future.isoformat()
                break
        result[sat_id] = {"next_comm_time": next_time}

    return {"observer": current_observer_name, "sim_time": ctx.sim_time.isoformat(), "next_comm": result}


@router.get("/api/iot_clusters/position", tags=["API/IoT"])
def get_iot_clusters_position(request: Request):
    """
    IoT 클러스터의 위치 정보를 반환하는 API
    """
    ctx = _ctx(request)

    # Topos는 직렬화 불가 → lat/lon만 반환
    clusters = {name: {"lat": t.latitude.degrees, "lon": t.longitude.degrees, "elev_m": ctx.raw_iot_clusters[name]["elevation_m"]}
                for name, t in ctx.iot_clusters.items()}
    return {"sim_time": ctx.sim_time.isoformat(), "clusters": clusters}


@router.get("/api/iot_clusters/visibility_schedule", tags=["API/IoT"])
def get_iot_clusters_visibility(request: Request, sat_id: int = Query(...)):
    """
    특정 위성이 IoT 클러스터에서 관측 가능한지 여부와 관측 가능 시간대를 반환하는 API
    """
    ctx = _ctx(request)

    if sat_id not in ctx.satellites:
        return {"error": f"sat_id {sat_id} not found"}
    satellite = ctx.satellites[sat_id]
    result = []

    for name, cluster in ctx.iot_clusters.items():
        visible_periods: List[tuple[str, str]] = []
        visible = False
        start: Optional[datetime] = None
        for offset in range(0, 7200, 30):
            future = ctx.sim_time + timedelta(seconds=offset)
            if elevation_deg(satellite, cluster, to_ts(future)) >= ctx.threshold_deg:
                if not visible:
                    start = future
                    visible = True
            else:
                if visible:
                    visible_periods.append((start.isoformat(), future.isoformat()))
                    visible = False
        if visible and start:
            visible_periods.append((start.isoformat(), future.isoformat()))
        if visible_periods:
            result.append({"iot_cluster": name, "periods": visible_periods})
    return {"sim_time": ctx.sim_time.isoformat(), "sat_id": sat_id, "schedule": result}


@router.get("/api/iot_clusters/visible", tags=["API/IoT"])
def get_iot_clusters_visible(
    request: Request,
    sat_id: Optional[int] = Query(None, description="위성 ID"),
    iot_name: Optional[str] = Query(None, description="IoT 클러스터 이름")
):
    """
    특정 IoT 클러스터에서 관측 가능한지 여부를 반환하는 API\n
    - sat_id만 지정 → 해당 위성과 통신 가능한 IoT 클러스터 목록 반환
    - iot_name만 지정 → 해당 IoT 클러스터와 통신 가능한 위성 ID 목록 반환
    """
    ctx = _ctx(request)

    if (sat_id is None) == (iot_name is None):
        return {"error": "neither sat_id or iot_name not found"}

    t_ts = get_current_time_utc(ctx.sim_time)

    if sat_id is not None:
        if sat_id not in ctx.satellites:
            return {"error": f"sat_id {sat_id} not found"}
        satellite = ctx.satellites[sat_id]
        visible_clusters = [name for name, cluster in ctx.iot_clusters.items()
                            if elevation_deg(satellite, cluster, t_ts) >= ctx.threshold_deg]
        return {"sat_id": sat_id, "sim_time": ctx.sim_time.isoformat(), "visible_iot_clusters": visible_clusters}

    cluster_name = iot_name
    if cluster_name not in ctx.iot_clusters:
        return {"error": f"iot cluster '{cluster_name}' not found"}
    cluster = ctx.iot_clusters[cluster_name]
    visible_sats = [sid for sid, sat in ctx.satellites.items()
                    if elevation_deg(sat, cluster, t_ts) >= ctx.threshold_deg]
    return {"sim_time": ctx.sim_time.isoformat(), "iot_cluster": cluster_name, "visible_sat_ids": visible_sats}


@router.get("/api/iot_clusters/visible_count", tags=["API/IoT"])
def get_iot_clusters_visible_count(request: Request, iot_name: str = Query(...)):
    """
    특정 IoT 클러스터에서 관측 가능한 위성의 개수를 반환하는 API
    """
    ctx = _ctx(request)

    if iot_name not in ctx.iot_clusters:
        return {"error": f"iot cluster '{iot_name}' not found"}
    cluster = ctx.iot_clusters[iot_name]
    t_ts = get_current_time_utc(ctx.sim_time)
    count = sum(1 for _, sat in ctx.satellites.items() if elevation_deg(sat, cluster, t_ts) >= ctx.threshold_deg)
    return {"sim_time": ctx.sim_time.isoformat(), "iot_cluster": iot_name, "visible_count": count}

@router.get("/api/data/summary", tags=["API/Data"])
def get_data_summary(
    request: Request,
    detail: bool = Query(False, description="위성(클라이언트)별 상세 목록 포함"),
    histogram: bool = Query(False, description="위성별 클래스 분포 포함(10 클래스)"),
    limit: int = Query(20, ge=1, le=1000, description="detail=false일 때 나열할 위성 수"),
):
    """
    CIFAR-10 분할/할당 상태 요약
    - 기본: 전역 설정/수량 요약만 반환
    - detail=true: 위성별 샘플 수 포함(기본 최대 limit개)
    - histogram=true: 위성별 클래스 히스토그램(0~9) 포함
    """
    ctx = _ctx(request)

    # 데이터 레지스트리/헬퍼 확인
    if "DATA_REGISTRY" not in globals() or "get_training_dataset" not in globals():
        return {"error": "DATA_REGISTRY / get_training_dataset 가 초기화되지 않았습니다."}

    # 구성값/환경
    try:
        cfg = {
            "samples_per_client": SAMPLES_PER_CLIENT,
            "dirichlet_alpha": DIRICHLET_ALPHA,
            "seed": RNG_SEED,
            "with_replacement": WITH_REPLACEMENT,
            "cifar_root": str(CIFAR_ROOT),
        }
    except Exception as e:
        cfg = {"error": f"config fetch failed: {e}"}

    # 위성/할당 현황
    sat_ids = sorted(ctx.satellites.keys())
    assignments = getattr(DATA_REGISTRY, "assignments", {})
    num_clients_assigned = len(assignments) if isinstance(assignments, dict) else len(sat_ids)

    result: Dict[str, Any] = {
        "dataset": "CIFAR-10",
        "config": cfg,
        "counts": {
            "num_satellites": len(sat_ids),
            "num_clients_assigned": num_clients_assigned,
        },
        "per_client": [],
        "notes": [
            "detail=true 로 전체 위성 별 샘플 수를 포함할 수 있습니다.",
            "histogram=true 로 (0~9) 클래스 분포를 포함할 수 있습니다.",
            "detail=false 인 경우 limit 만큼만 나열합니다.",
        ],
    }

    # 상세가 아니면 샘플 수 집계를 건너뛰고 끝
    if not detail and not histogram:
        return result

    # 상세 목록 만들기 (detail=false면 limit 만큼만)
    target_ids = sat_ids if detail else sat_ids[:limit]
    total_listed_samples = 0

    for sid in target_ids:
        try:
            ds = get_training_dataset(sid)  # TensorDataset 예상
            n = len(ds) if hasattr(ds, "__len__") else None
            entry: Dict[str, Any] = {"sat_id": sid, "num_samples": n}
            total_listed_samples += (n or 0)

            if histogram and n and hasattr(ds, "tensors") and len(ds.tensors) >= 2:
                labels = ds.tensors[1]
                hist: Dict[int, int] = {}
                try:
                    # torch가 있다면 torch로
                    if ctx.features and ctx.features.has_torch:
                        import torch
                        uniq, cnt = torch.unique(labels.cpu(), return_counts=True)
                        hist = {int(u.item()): int(c.item()) for u, c in zip(uniq, cnt)}
                    else:
                        # torch 미사용 대비
                        arr = labels.detach().cpu().numpy()
                        c = Counter(arr.tolist())
                        hist = {int(k): int(v) for k, v in c.items()}
                except Exception as e:
                    entry["hist_error"] = f"{e}"
                else:
                    # 0~9 클래스 누락된 항목 0으로 채우기(보기 좋게)
                    for cls in range(10):
                        hist.setdefault(cls, 0)
                    entry["class_hist"] = hist
            result["per_client"].append(entry)
        except Exception as e:
            result["per_client"].append({"sat_id": sid, "error": str(e)})

    # 참고 합계(나열된 항목 기준)
    result["counts"]["total_samples_listed"] = total_listed_samples
    if not detail:
        result["counts"]["listed_clients"] = len(target_ids)

    return result

@router.get("/api/model/global", tags=["API/Model"])
def get_global_model_info(request: Request, detail: bool = Query(False, description="모든 글로벌 체크포인트 목록/메타데이터 포함")):
    """
    글로벌 모델의 현재 버전/경로를 조회합니다.
    - version: GLOBAL_MODEL_VERSION (초기화 전이면 -1)
    - latest_ckpt: CKPT_DIR/global_v{version}.ckpt 가 존재하면 그 경로를, 없으면 가장 최근 파일을 찾아 반환
    - detail=true 일 때: CKPT_DIR 안의 global_v*.ckpt 파일 리스트와 파일 메타(크기, mtime)를 함께 반환
    """
    ctx = _ctx(request)

    # 안전장치: 필요한 전역이 없을 수 있을 때 기본값
    version = globals().get("GLOBAL_MODEL_VERSION", -1)
    ckpt_dir = globals().get("CKPT_DIR", Path(__file__).parent / "ckpt")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # 우선 canonical 경로(global_v{ver}.ckpt)를 시도
    canonical = ckpt_dir / f"global_v{version}.ckpt" if version >= 0 else None
    latest_path = canonical if (canonical and canonical.exists()) else None

    # 없으면 가장 최신 파일을 스캔
    if latest_path is None:
        # _find_latest_global_ckpt()가 있다면 활용
        if "find_latest_global_ckpt" in globals():
            try:
                best_ver, best_path = find_latest_global_ckpt()
                if best_ver is not None and best_path is not None and best_path.exists():
                    version = max(version, best_ver)
                    latest_path = best_path
            except Exception:
                pass
        # 그래도 못 찾으면 패턴 매칭으로 스캔
        if latest_path is None:
            candidates = sorted(ckpt_dir.glob("global_v*.ckpt"))
            if candidates:
                latest_path = candidates[-1]

    resp = {
        "version": int(version),
        "latest_ckpt": str(latest_path) if latest_path else None,
        "initialized": bool(version >= 0 and latest_path is not None),
    }

    if detail:
        files = []
        for p in sorted(ckpt_dir.glob("global_v*.ckpt")):
            try:
                stat = p.stat()
                files.append({
                    "path": str(p),
                    "size_bytes": stat.st_size,
                    "mtime_epoch": int(stat.st_mtime),
                    "mtime_iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(stat.st_mtime)),
                })
            except Exception:
                files.append({"path": str(p)})
        resp["files"] = files

    return resp


@router.get("/api/model/global/version", tags=["API/Model"])
def get_global_model_version():
    """
    글로벌 모델 버전 숫자만 간단히 반환합니다.
    """
    version = globals().get("GLOBAL_MODEL_VERSION", -1)
    return {"version": int(version)}


@router.get("/api/agg_stats")
def agg_stats(request: Request, limit: int = 50):
    ctx = request.app.state.ctx
    items = list(getattr(ctx, "AGG_LOG", []))
    return {
        "global_version": int(ctx.GLOBAL_MODEL_VERSION),
        "items": items[-int(limit):]
    }