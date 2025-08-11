# sat_comm_server.py
from __future__ import annotations

import os
import json
import asyncio
import threading
import logging
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, Future
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from collections import Counter

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse
from skyfield.api import load, EarthSatellite, Topos
from pathlib import Path

# --- data registry (로컬 CIFAR 분배 모듈) ---
from data import (
    CIFAR_ROOT, ASSIGNMENTS_DIR, SAMPLES_PER_CLIENT, DIRICHLET_ALPHA, RNG_SEED, WITH_REPLACEMENT,
    DATA_REGISTRY, get_training_dataset,
)

# -------------------- Optional: torch for GPU detection --------------------
try:
    import torch
    _has_torch = True
except Exception:
    _has_torch = False

# 체크포인트 디렉토리: simserver/ckpt
BASE_DIR = Path(__file__).resolve().parent
CKPT_DIR = BASE_DIR / "ckpt"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

# -------------------- Logger --------------------
logger = logging.getLogger("simserver")
logger.setLevel(logging.INFO)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(h)


# ==================== 시뮬레이션 상태 변수 ====================
satellites: Dict[int, EarthSatellite] = {}         # sat_id -> EarthSatellite
sat_comm_status: Dict[int, bool] = {}              # sat_id -> 통신 가능 여부
current_sat_positions: Dict[int, Dict[str, float]] = {}

observer_locations = {
    "Berlin":  Topos(latitude_degrees=52.52,  longitude_degrees=13.41,  elevation_m=34),
    "Houston": Topos(latitude_degrees=29.76,  longitude_degrees=-95.37, elevation_m=30),
    "Tokyo":   Topos(latitude_degrees=35.68,  longitude_degrees=139.69, elevation_m=40),
    "Nairobi": Topos(latitude_degrees=-1.29,  longitude_degrees=36.82,  elevation_m=1700),
    "Sydney":  Topos(latitude_degrees=-33.87, longitude_degrees=151.21, elevation_m=58),
}
raw_iot_clusters = {
    "Abisko":         {"latitude": 68.35,  "longitude": 18.79,  "elevation_m": 420},
    "Boreal":         {"latitude": 55.50,  "longitude": 105.00, "elevation_m": 450},
    "Taiga":          {"latitude": 58.00,  "longitude": 99.00,  "elevation_m": 300},
    "Patagonia":      {"latitude": 51.00,  "longitude": 73.00,  "elevation_m": 500},
    "Amazon_Forest":  {"latitude": -3.47,  "longitude": -62.37, "elevation_m": 100},
    "Great_Barrier":  {"latitude": -18.29, "longitude": 147.77, "elevation_m": 0},
    "Mediterranean":  {"latitude": 37.98,  "longitude": 23.73,  "elevation_m": 170},
    "California":     {"latitude": 36.78,  "longitude": -119.42,"elevation_m": 150},
}
iot_clusters = {
    name: Topos(latitude_degrees=cfg["latitude"], longitude_degrees=cfg["longitude"], elevation_m=cfg["elevation_m"])
    for name, cfg in raw_iot_clusters.items()
}

current_observer_name = "Berlin"
observer = observer_locations[current_observer_name]

ts = load.timescale()
sim_time = datetime(2025, 3, 30, 0, 0, 0)  # 시뮬레이션 시작 시간
threshold_deg = 40
sim_paused = False
auto_resume_delay_sec = 0
sim_delta_sec = 1.0        # 시뮬레이션 한 스텝 증가량(초)
real_interval_sec = 0.05   # 실제 루프 슬립(초)


def to_ts(dt: datetime):
    """datetime -> skyfield ts.utc(...)"""
    return ts.utc(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second)


def get_current_time_utc():
    return to_ts(sim_time)


def elevation_deg(sat: EarthSatellite, topox: Topos, t_ts):
    """위성-관측점 고도(deg) 계산."""
    alt, _, _ = (sat - topox).at(t_ts).altaz()
    return alt.degrees


# ==================== 연합학습(FL) 런타임 상태 ====================
# GPU 개수 자동/환경 지정
NUM_GPUS = int(os.getenv("NUM_GPUS", "0"))
if NUM_GPUS == 0 and _has_torch and torch.cuda.is_available():
    NUM_GPUS = torch.cuda.device_count()

# GPU당 동시 세션 수(기본 1)
SESSIONS_PER_GPU = int(os.getenv("SESSIONS_PER_GPU", "6"))

# 총 동시 학습 작업 수
MAX_TRAIN_WORKERS = int(os.getenv(
    "FL_MAX_WORKERS",
    str(max(1, (NUM_GPUS or 1) * max(1, SESSIONS_PER_GPU))))
)

training_executor = ThreadPoolExecutor(max_workers=MAX_TRAIN_WORKERS)
uploader_executor = ThreadPoolExecutor(max_workers=4)  # 업로드/집계는 짧게

# GPU별 세마포어
gpu_ids = list(range(NUM_GPUS)) if NUM_GPUS > 0 else [None]  # None = CPU fallback
gpu_semaphores = {gid: threading.BoundedSemaphore(SESSIONS_PER_GPU)
                  for gid in gpu_ids if gid is not None}
_gpu_rr_idx = 0
_gpu_rr_lock = threading.Lock()

train_queue: "asyncio.Queue[int]" = asyncio.Queue(
    maxsize=int(os.getenv("FL_QUEUE_MAX", "1000"))
)

def _log(msg: str):
    logger.info(f"[FL] {msg}")


def try_pick_gpu_id_nonblocking():
    """
    빈 세션 있는 GPU만 즉시 할당. 없으면 (None, None)
    """
    global _gpu_rr_idx
    if gpu_ids == [None]:
        return None, None
    with _gpu_rr_lock:
        start = _gpu_rr_idx
        n = len(gpu_ids)
        for i in range(n):
            gid = gpu_ids[(start + i) % n]
            sem = gpu_semaphores[gid]
            if sem.acquire(blocking=False):
                _gpu_rr_idx = (start + i + 1) % n
                return gid, sem
    return None, None


@dataclass
class TrainState:
    running: bool = False
    future: Optional[Future] = None
    gpu_id: Optional[int] = None
    stop_event: threading.Event = field(default_factory=threading.Event)
    last_ckpt_path: Optional[str] = None
    round_idx: int = 0  # False→True마다 라운드 증가
    in_queue: bool = False


train_states: Dict[int, TrainState] = {}

def _enqueue_training(sat_id: int):
    st = train_states[sat_id]
    if st.running or st.in_queue:
        return
    try:
        train_queue.put_nowait(sat_id)
        st.in_queue = True
        # _log(f"SAT{sat_id}: queued")  # 필요하면 주석 해제
    except asyncio.QueueFull:
        # 너무 많은 요청이면 조용히 드롭(또는 드물게만 로그)
        pass

def _build_worker_gpu_list():
    # 예: NUM_GPUS=2, SESSIONS_PER_GPU=2 -> [0,0,1,1]
    if NUM_GPUS > 0:
        return [gid for gid in range(NUM_GPUS) for _ in range(SESSIONS_PER_GPU)]
    # GPU 없을 때도 워커는 필요(=CPU 학습)
    return [None] * max(1, SESSIONS_PER_GPU)

async def _train_worker(gpu_id: Optional[int]):
    loop = asyncio.get_running_loop()
    while True:
        sat_id = await train_queue.get()
        st = train_states[sat_id]
        st.in_queue = False

        # 이미 다른 곳에서 시작되었으면 skip
        if st.running:
            train_queue.task_done()
            continue

        st.stop_event.clear()
        st.gpu_id = gpu_id
        st.running = True

        def _job():
            # do_local_training 안에서 gpu_id로 디바이스 선택
            return do_local_training(sat_id=sat_id, stop_event=st.stop_event, gpu_id=gpu_id)

        # 스레드풀에서 학습 실행 (이벤트루프 블로킹 방지)
        fut = loop.run_in_executor(training_executor, _job)
        st.future = fut  # asyncio.Future

        try:
            ckpt = await fut
        except Exception as e:
            _log(f"SAT{sat_id}: training ERROR: {e}")
            ckpt = None
        finally:
            st.last_ckpt_path = ckpt
            st.running = False
            _log(f"SAT{sat_id}: training DONE (worker gpu={gpu_id}), ckpt={ckpt}")
            train_queue.task_done()

def _upload_and_aggregate_async(sat_id: int, ckpt_path: Optional[str]):
    """업로드/집계 비동기 실행(사용자 훅 호출)."""
    if not ckpt_path:
        _log(f"SAT{sat_id}: no checkpoint to upload")
        return

    def _job():
        _log(f"SAT{sat_id}: uploading {ckpt_path}")
        try:
            if "upload_model_to_server" in globals():
                upload_model_to_server(sat_id=sat_id, ckpt_path=ckpt_path)
                _log(f"SAT{sat_id}: uploaded")
            if "upload_and_aggregate" in globals():
                new_global = upload_and_aggregate(sat_id, ckpt_path)
                _log(f"SAT{sat_id}: aggregated -> {new_global}")
        except Exception as e:
            _log(f"SAT{sat_id}: upload/aggregate ERROR: {e}")

    uploader_executor.submit(_job)


def _launch_training(sat_id: int):
    """오프라인일 때 학습 시작(중복 방지, GPU 슬롯 없으면 다음 루프에서 재시도)."""
    st = train_states[sat_id]
    if st.running and st.future and not st.future.done():
        return

    gid, pre_acquired_sem = try_pick_gpu_id_nonblocking()
    if gid is None and NUM_GPUS > 0:
        _log(f"SAT{sat_id}: GPUs busy, will retry later")
        return

    st.stop_event = threading.Event()
    st.gpu_id = gid
    st.running = True

    def _job():
        _log(f"SAT{sat_id}: training START (gpu={st.gpu_id})")
        ckpt = None
        try:
            if _has_torch and st.gpu_id is not None:
                torch.cuda.set_device(st.gpu_id)
            ckpt = do_local_training(sat_id=sat_id, stop_event=st.stop_event, gpu_id=st.gpu_id)
        except Exception as e:
            _log(f"SAT{sat_id}: training ERROR: {e}")
        finally:
            st.last_ckpt_path = ckpt
            st.running = False
            _log(f"SAT{sat_id}: training DONE, ckpt={ckpt}")
            if pre_acquired_sem is not None:
                pre_acquired_sem.release()

    st.future = training_executor.submit(_job)

def _on_become_visible(sat_id: int):
    st = train_states[sat_id]
    if st.running and st.future:
        st.stop_event.set()
        def _cb(_fut):
            try:
                _fut.result()  # 예외 전파/소거
            except Exception:
                pass
            _upload_and_aggregate_async(sat_id, train_states[sat_id].last_ckpt_path)
        # asyncio.Future / concurrent.futures.Future 둘 다 지원
        try:
            st.future.add_done_callback(_cb)
        except Exception:
            # 일부 구현에서 add_done_callback 시그니처 차이 처리
            pass
    else:
        _upload_and_aggregate_async(sat_id, st.last_ckpt_path)


# === 로컬 학습 함수: do_local_training ===
def do_local_training(
    sat_id: int,
    stop_event: threading.Event,
    gpu_id: Optional[int] = None,
    *,
    epochs: Optional[int] = None,
    lr: Optional[float] = None,
    batch_size: Optional[int] = None,
) -> str:
    """
    로컬 학습을 수행하고 마지막 체크포인트 파일 경로를 반환.
    - stop_event가 set되면 가능한 빠르게 중단.
    - gpu_id가 주어지면 해당 GPU를 사용.
    - 데이터는 data.get_training_dataset(sat_id)에서 획득.
    - 글로벌 초기 가중치 훅(get_initial_model_state)이 있으면 적용.
    """
    import tempfile
    import datetime as _dt
    from torch.utils.data import DataLoader
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torchvision.models import mobilenet_v3_small

    # ---- 하이퍼파라미터: 환경변수 -> 인자 -> 기본값 ----
    EPOCHS = int(os.getenv("FL_EPOCHS_PER_ROUND", "10")) if epochs is None else int(epochs)
    LR     = float(os.getenv("FL_LR", "1e-3"))          if lr is None else float(lr)
    BS     = int(os.getenv("FL_BATCH_SIZE", "64"))      if batch_size is None else int(batch_size)
    NUM_CLASSES = int(os.getenv("FL_NUM_CLASSES", "10"))

    # ---- 디바이스 선정 ----
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_id}") if gpu_id is not None else torch.device("cuda")
    else:
        device = torch.device("cpu")

    # ---- 모델 준비 ----
    model = mobilenet_v3_small(num_classes=NUM_CLASSES)
    if "get_initial_model_state" in globals():
        try:
            state_dict = get_initial_model_state(sat_id)
            if state_dict is not None:
                model.load_state_dict(state_dict)
        except Exception as e:
            _log(f"SAT{sat_id}: failed to load initial model state: {e}")
    model.to(device)
    model.train()

    # ---- 데이터 준비 (반드시 data 레지스트리 사용) ----
    dataset = get_training_dataset(sat_id)
    loader = DataLoader(
        dataset, batch_size=BS, shuffle=True, drop_last=False,
        num_workers=2, pin_memory=torch.cuda.is_available()
    )

    # ---- 옵티마이저/로스 ----
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    def save_ckpt(ep: int) -> str:
        import datetime as _dt
        ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"sat{sat_id}_round{train_states[sat_id].round_idx}_ep{ep}_{ts}.ckpt"
        ckpt_path = CKPT_DIR / fname
        torch.save(model.state_dict(), ckpt_path)
        return str(ckpt_path)

    last_ckpt = None

    for ep in range(EPOCHS):
        if stop_event.is_set():
            break

        running_loss, correct, total = 0.0, 0, 0
        for images, labels in loader:
            if stop_event.is_set():
                break
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
            _, pred = outputs.max(1)
            total += labels.size(0)
            correct += int(pred.eq(labels).sum().item())

        avg_loss = running_loss / max(1, len(loader))
        acc = 100.0 * correct / max(1, total)
        last_ckpt = save_ckpt(ep)

        _log(f"SAT{sat_id}: ep={ep+1}/{EPOCHS} loss={avg_loss:.4f} acc={acc:.2f}% saved={last_ckpt}")

    return last_ckpt or save_ckpt(-1)


# ==================== FastAPI 앱/수명주기 ====================
@asynccontextmanager
async def lifespan(app: FastAPI):
    await initialize_simulation()
    try:
        yield
    finally:
        # executors 정리
        training_executor.shutdown(wait=False, cancel_futures=True)
        uploader_executor.shutdown(wait=False, cancel_futures=True)


app = FastAPI(
    title="Satellite Communication API",
    description="위성, 지상국, IoT 클러스터 간 통신 상태 및 관측 가능 시간 등을 제공하는 API 서비스입니다.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)


# ==================== 서버 기동 시 처리 ====================
async def initialize_simulation():
    global satellites

    tle_path = "../constellation.tle"
    if not os.path.exists(tle_path):
        raise FileNotFoundError(f"TLE 파일이 존재하지 않습니다: {tle_path}")

    with open(tle_path, "r") as f:
        lines = [line.strip() for line in f.readlines()]
        for i in range(0, len(lines), 3):
            name, line1, line2 = lines[i:i+3]
            sat_id = int(name.replace("SAT", ""))
            satellite = EarthSatellite(line1, line2, name, ts)
            satellites[sat_id] = satellite

    # 초기 위치 계산 & FL 상태 초기화
    t = get_current_time_utc()
    for sat_id, satellite in satellites.items():
        subpoint = satellite.at(t).subpoint()
        current_sat_positions[sat_id] = {
            "lat": subpoint.latitude.degrees,
            "lon": subpoint.longitude.degrees,
        }
        sat_comm_status[sat_id] = False
        train_states[sat_id] = TrainState()

    # --- CIFAR-10 로드 & 위성별 가상 데이터셋 배정 ---
    try:
        DATA_REGISTRY.load_base(root=CIFAR_ROOT, seed=RNG_SEED, download=True)
        assign_file = ASSIGNMENTS_DIR / f"assign_seed{RNG_SEED}_alpha{DIRICHLET_ALPHA}_spc{SAMPLES_PER_CLIENT}.npz"
        if not DATA_REGISTRY.load_assignments(assign_file):
            sat_ids_sorted = sorted(satellites.keys())
            DATA_REGISTRY.assign_clients(
                sat_ids=sat_ids_sorted,
                samples_per_client=SAMPLES_PER_CLIENT,
                alpha=DIRICHLET_ALPHA,
                seed=RNG_SEED,
                with_replacement=WITH_REPLACEMENT,
            )
            DATA_REGISTRY.save_assignments(assign_file)
        logger.info(f"[DATA] CIFAR ready. per_client={SAMPLES_PER_CLIENT}, alpha={DIRICHLET_ALPHA}, seed={RNG_SEED}")
    except Exception as e:
        logger.error(f"[DATA] CIFAR init failed: {e}")

    for gid in _build_worker_gpu_list():
        asyncio.create_task(_train_worker(gid))

    asyncio.create_task(simulation_loop())


# ==================== 시뮬레이션 루프 ====================
async def simulation_loop():
    global sim_time, current_sat_positions

    while True:
        if not sim_paused:
            t_ts = get_current_time_utc()
            current_sat_positions = {}

            for sat_id, satellite in satellites.items():
                # 이전 가시성
                prev_visible = bool(sat_comm_status.get(sat_id, False))

                # 현재 가시성 계산
                alt_deg = elevation_deg(satellite, observer, t_ts)
                visible_now = (alt_deg >= threshold_deg)
                sat_comm_status[sat_id] = visible_now

                # 위치 업데이트
                subpoint = satellite.at(t_ts).subpoint()
                current_sat_positions[sat_id] = {
                    "lat": subpoint.latitude.degrees,
                    "lon": subpoint.longitude.degrees,
                }

                # 연합학습 트리거
                if not visible_now:
                    # 오프라인 → 학습 시작(가능 시)
                    _enqueue_training(sat_id)
                elif (not prev_visible) and visible_now:
                    # False -> True 전이
                    _on_become_visible(sat_id)
                    train_states[sat_id].round_idx += 1

            sim_time += timedelta(seconds=sim_delta_sec)

        await asyncio.sleep(real_interval_sec)


# ==================== 대시보드 / 페이지 ====================
@app.get("/dashboard", response_class=HTMLResponse, tags=["PAGE"])
def dashboard():
    paused_status = "Paused" if sim_paused else "Running"
    return f"""
    <html>
    <head>
        <title>Satellite Communication Dashboard</title>
        <meta http-equiv="refresh" content="5">
        <style>
            body {{ font-family: Arial; background: #f2f2f2; margin: 2em; }}
            h1 {{ color: #333; }}
            a {{ display: block; margin: 10px 0; font-size: 1.2em; }}
        </style>
    </head>
    <body>
        <h1>🛰️🛰 Satellite Communication Dashboard</h1>
        <p><b>Sim Time:</b> {sim_time.isoformat()}</p>
        <p><b>Status:</b> {paused_status}</p>
        <p><b>Step:</b> Δt={sim_delta_sec}s, Interval={real_interval_sec}s</p>
        <div class="control-form">
            <label>Δt(초): <input id="delta" type="number" step="any" value="{sim_delta_sec}" /></label>
            <label>간격(초): <input id="interval" type="number" step="any" value="{real_interval_sec}" /></label>
            <button onclick="setStep()">적용</button>
        </div>
        <script>
        async function setStep() {{
            const d = document.getElementById('delta').value;
            const i = document.getElementById('interval').value;
            const res = await fetch(`/api/set_step?delta_sec=${{d}}&interval_sec=${{i}}`, {{ method: 'PUT' }});
            const data = await res.json();
            if (!data.error) {{
                alert(`설정 완료: Δt=${{data.sim_delta_sec}}, Interval=${{data.real_interval_sec}}`);
                window.location.reload();
            }} else {{
                alert(`오류: ${{data.error}}`);
            }}
        }}
        </script>
        <hr>
        <a href="/gs_visibility">🛰️ GS별 통신 가능 위성 보기</a>
        <a href="/orbit_paths/lists">🛰 위성별 궤적 경로 보기</a>
        <a href="/map_path">🗺 지도 기반 위성 경로 보기</a>
        <a href="/visibility_schedules/lists">📅 위성별 관측 가능 시간대 목록 보기</a>
        <a href="/iot_clusters"> 📡 IoT 클러스터별 위치 보기</a>
        <a href="/iot_visibility"> 🌐 IoT 클러스터별 통신 가능 위성 보기</a>
        <a href="/comm_targets/lists">🚀 위성 통신 대상 확인</a>
    </body>
    </html>
    """


@app.get("/gs_visibility", response_class=HTMLResponse, tags=["PAGE"])
def gs_visibility():
    paused_status = "Paused" if sim_paused else "Running"
    gs_sections = []
    t_ts = get_current_time_utc()

    for name, gs in observer_locations.items():
        rows = []
        for sid, sat in satellites.items():
            alt_deg = elevation_deg(sat, gs, t_ts)
            if alt_deg >= threshold_deg:
                rows.append(f'<tr><td>{sid}</td><td>{alt_deg:.2f}°</td></tr>')
        table_html = f"""
        <h2>{name}</h2>
        <table>
            <tr><th>Sat ID</th><th>Elevation</th></tr>
            {''.join(rows)}
        </table>
        """
        gs_sections.append(table_html)

    return f"""
    <html>
    <head>
        <title>GS Visibility</title>
        <meta http-equiv="refresh" content="5">
        <style>
            body {{ font-family: Arial; background: #f2f2f2; margin: 2em; }}
            h2 {{ margin-top: 2em; }}
            table {{ border-collapse: collapse; width: 60%; margin-bottom: 2em; }}
            th, td {{ border: 1px solid #ccc; padding: 8px; text-align: center; }}
        </style>
    </head>
    <body>
        <p><a href="/dashboard">← Back to Dashboard</a></p>
        <h1>🛰️ GS-wise Visible Satellites</h1>
        <p><b>Sim Time:</b> {sim_time.isoformat()}</p>
        <p><b>Status:</b> {paused_status}</p>
        <p><b>Step:</b> Δt={sim_delta_sec}s, Interval={real_interval_sec}s</p>
        <hr>
        {''.join(gs_sections)}
    </body>
    </html>
    """


@app.get("/orbit_paths/lists", response_class=HTMLResponse, tags=["PAGE"])
def sat_paths():
    links = [f'<li><a href="/orbit_paths?sat_id={sid}">SAT{sid} Path</a></li>' for sid in sorted(satellites)]
    return f"""
    <html>
    <head>
        <title>Satellite Paths</title>
        <style>
            body {{ font-family: Arial; margin: 2em; }}
            ul {{ list-style-type: none; padding: 0; }}
            li {{ margin-bottom: 10px; }}
        </style>
    </head>
    <body>
        <p><a href="/dashboard">← Back to Dashboard</a></p>
        <h1>🛰 All Satellite Orbit Paths</h1>
        <ul>
            {''.join(links)}
        </ul>
    </body>
    </html>
    """


@app.get("/orbit_paths", response_class=HTMLResponse, tags=["PAGE"])
def orbit_paths(sat_id: int = Query(...)):
    if sat_id not in satellites:
        return HTMLResponse(f"<p>Error: sat_id {sat_id} not found</p>", status_code=404)

    satellite = satellites[sat_id]
    t0 = sim_time
    positions = []
    for offset_sec in range(0, 7200, 60):
        future = t0 + timedelta(seconds=offset_sec)
        t_ts = to_ts(future)
        subpoint = satellite.at(t_ts).subpoint()
        positions.append((subpoint.latitude.degrees, subpoint.longitude.degrees))

    rows = ''.join([f'<tr><td>{i*60}s</td><td>{lat:.2f}</td><td>{lon:.2f}</td></tr>'
                    for i, (lat, lon) in enumerate(positions)])

    return f"""
    <html>
    <head>
        <title>Orbit Track for SAT{sat_id}</title>
        <style>
            body {{ font-family: Arial; margin: 2em; }}
            table {{ border-collapse: collapse; width: 60%; }}
            th, td {{ border: 1px solid #ccc; padding: 8px; text-align: center; }}
        </style>
    </head>
    <body>
        <p><a href="/orbit_paths/lists">← Back to All Satellite Orbit Paths</a></p>
        <h1>🛰 SAT{sat_id} Orbit Path</h1>
        <table>
            <tr><th>Offset</th><th>Latitude</th><th>Longitude</th></tr>
            {rows}
        </table>
    </body>
    </html>
    """


@app.get("/map_path", response_class=HTMLResponse, tags=["PAGE"])
def map_path():
    options = ''.join(f'<option value="{sid}">SAT{sid}</option>' for sid in sorted(satellites))
    obs_data_json = json.dumps({name: {"lat": gs.latitude.degrees, "lon": gs.longitude.degrees}
                                for name, gs in observer_locations.items()})
    iot_data_json = json.dumps(raw_iot_clusters)
    return f"""
    <html>
    <head>
        <title>Map Path</title>
        <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
        <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
        <style>
            #map {{ height: 90vh; }}
            html, body {{ margin: 0; padding: 0; }}
            .coverage-circle {{ fill-opacity:0.5; stroke-width:0; }}
            .leaflet-tooltip.no-box {{
                background: transparent; border: none; box-shadow: none; padding: 0; font-weight: bold;
            }}
            .leaflet-tooltip.no-box::before {{ display: none; }}
        </style>
    </head>
    <body>
        <p><a href="/dashboard">← Back to Dashboard</a></p>
        <h1>🗺 Satellite Map Path</h1>
        <div id="sim-time"></div>
        <div id="sim-step" style="margin-bottom: 1em;"></div>
        <label for="sat_id">Choose a satellite:</label>
        <select id="sat_id" onchange="drawTrajectory(this.value)">{options}</select>
        <div id="map"></div>
        <script>
            var map = L.map('map', {{
                center: [0, 0],
                zoom: 2,
                worldCopyJump: false,
                maxBounds: [[-85, -180], [85, 180]],
                maxBoundsViscosity: 1.0,
                inertia: false
            }});
            L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png').addTo(map);

            const observers = {obs_data_json};
            const iotClusters = {iot_data_json};

            var circles = [];
            var pathLines = [];
            var currentMarker = null;
            var currentLabel = null;
            var markerInterval = null;
            var coverCircles = [];

            function drawCoverage() {{
                for (let [name, loc] of Object.entries(observers)) {{
                    let c = L.circle([loc.lat, loc.lon], {{
                        radius: 100000, color: 'green', fillColor: 'green', className: 'coverage-circle'
                    }}).addTo(map);
                    c.bindTooltip(name, {{ permanent: true, direction: 'center', className: 'no-box' }});
                    coverCircles.push(c);
                }}
                for (let [name, loc] of Object.entries(iotClusters)) {{
                    let c = L.circle([loc.latitude, loc.longitude], {{
                        radius: 50000, color: 'orange', fillColor: 'orange', className: 'coverage-circle'
                    }}).addTo(map);
                    c.bindTooltip(name, {{ permanent: true, direction: 'center', className: 'no-box' }});
                    coverCircles.push(c);
                }}
            }}

            async function drawTrajectory(sat_id) {{
                circles.forEach(c => map.removeLayer(c)); circles = [];
                pathLines.forEach(p => map.removeLayer(p)); pathLines = [];
                if (currentMarker) map.removeLayer(currentMarker);
                if (currentLabel) map.removeLayer(currentLabel);
                if (markerInterval) clearInterval(markerInterval);

                drawCoverage();

                let resp = await fetch(`/api/trajectory?sat_id=${{sat_id}}`);
                let satData = await resp.json();
                for (let segment of satData.segments) {{
                    let latlngs = [];
                    for (let p of segment) {{
                        let latlng = [p.lat, p.lon];
                        latlngs.push(latlng);
                        let marker = L.circleMarker(latlng, {{radius: 1, color: 'red'}}).addTo(map);
                        circles.push(marker);
                    }}
                    let line = L.polyline(latlngs, {{color: 'blue', weight: 1}}).addTo(map);
                    pathLines.push(line);
                }}
                markerInterval = setInterval(async () => {{
                    let live = await fetch(`/api/position?sat_id=${{sat_id}}`);
                    let data = await live.json();
                    let simResp = await fetch(`/api/sim_time`);
                    let simData = await simResp.json();
                    document.getElementById('sim-time').innerHTML = `<p><b>Sim Time:</b> ${{simData.sim_time}}</p>`;
                    document.getElementById('sim-step').innerHTML = `<p><b>Step:</b> Δt={sim_delta_sec}s, Interval={real_interval_sec}s</p>`;

                    if (data.lat !== undefined && data.lon !== undefined) {{
                        if (currentMarker) map.removeLayer(currentMarker);
                        if (currentLabel) map.removeLayer(currentLabel);
                        currentMarker = L.circleMarker([data.lat, data.lon], {{radius: 3, color: 'blue'}}).addTo(map);
                        currentLabel = L.marker([data.lat, data.lon], {{
                            icon: L.divIcon({{
                                className: 'current-label',
                                html: '<b>현재 위성 위치</b>',
                                iconSize: [120, 20],
                                iconAnchor: [60, -10]
                            }})
                        }}).addTo(map);
                    }}
                }}, 1000);
            }}

            window.onload = () => {{
                const selector = document.getElementById('sat_id');
                if (selector.value) drawTrajectory(selector.value);
            }}
        </script>
    </body>
    </html>
    """


# ==================== API ====================
@app.post("/api/reset_time", tags=["API"])
def reset_sim_time():
    global sim_time
    sim_time = datetime(2025, 3, 30, 0, 0, 0)
    return {"status": "reset", "sim_time": sim_time.isoformat()}


@app.get("/api/sim_time", tags=["API"])
def get_sim_time_api():
    return {"sim_time": sim_time.isoformat()}


@app.put("/api/sim_time", tags=["API"])
def set_sim_time(year: int, month: int, day: int, hour: int = 0, minute: int = 0, second: int = 0):
    global sim_time
    try:
        sim_time = datetime(year, month, day, hour, minute, second)
        return {"status": "updated", "sim_time": sim_time.isoformat()}
    except Exception as e:
        return {"error": str(e)}


@app.put("/api/set_step", tags=["API"])
def set_step(delta_sec: float = Query(...), interval_sec: float = Query(...)):
    global sim_delta_sec, real_interval_sec
    if delta_sec <= 0 or interval_sec < 0:
        return {"error": "delta_sec은 양수, interval_sec은 0 이상이어야 합니다."}
    sim_delta_sec = float(delta_sec)
    real_interval_sec = float(interval_sec)
    return {"sim_delta_sec": sim_delta_sec, "real_interval_sec": real_interval_sec}


@app.post("/api/pause", tags=["API"])
def pause_simulation():
    global sim_paused
    sim_paused = True
    return {"status": "paused"}


@app.post("/api/resume", tags=["API"])
def resume_simulation():
    global sim_paused
    sim_paused = False
    return {"status": "resumed"}


@app.get("/api/trajectory", tags=["API"])
def get_trajectory(sat_id: int = Query(...)):
    if sat_id not in satellites:
        return {"error": f"sat_id {sat_id} not found"}
    satellite = satellites[sat_id]
    t0 = sim_time
    prev_lon = None
    segment: List[Dict[str, float]] = []
    segments: List[List[Dict[str, float]]] = []

    for offset_sec in range(0, 7200, 30):
        future = t0 + timedelta(seconds=offset_sec)
        t_ts = to_ts(future)
        subpoint = satellite.at(t_ts).subpoint()
        lat = subpoint.latitude.degrees
        lon = subpoint.longitude.degrees
        if prev_lon is not None and abs(lon - prev_lon) > 180:
            segments.append(segment)
            segment = []
        segment.append({"lat": lat, "lon": lon})
        prev_lon = lon

    if segment:
        segments.append(segment)
    return {"sat_id": sat_id, "segments": segments}


@app.get("/api/position", tags=["API"])
def get_position(sat_id: int = Query(...)):
    if sat_id not in current_sat_positions:
        return {"error": f"Position for SAT{sat_id} not available"}
    return current_sat_positions[sat_id]


@app.get("/api/comm_targets", tags=["API"])
def get_comm_targets(sat_id: int = Query(..., description="위성 ID")):
    if sat_id not in satellites:
        return {"error": f"sat_id {sat_id} not found"}
    t_ts = get_current_time_utc()
    sat = satellites[sat_id]
    visible_ground = [name for name, gs in observer_locations.items()
                      if elevation_deg(sat, gs, t_ts) >= threshold_deg]
    visible_iot = [name for name, cluster in iot_clusters.items()
                   if elevation_deg(sat, cluster, t_ts) >= threshold_deg]
    return {
        "sim_time": sim_time.isoformat(),
        "sat_id": sat_id,
        "visible_ground_stations": visible_ground,
        "visible_iot_clusters": visible_iot,
    }


@app.get("/api/gs/visibility", tags=["API/GS"])
def get_gs_visibility():
    result = {}
    t_ts = get_current_time_utc()
    for name, gs in observer_locations.items():
        visible_sats = []
        for sid, sat in satellites.items():
            alt_deg = elevation_deg(sat, gs, t_ts)
            if alt_deg >= threshold_deg:
                visible_sats.append({"sat_id": sid, "elevation": alt_deg})
        result[name] = visible_sats
    return {"sim_time": sim_time.isoformat(), "data": result}


@app.get("/api/gs/visibility_schedule", tags=["API/GS"])
def get_visibility_schedule(sat_id: int = Query(...)):
    if sat_id not in satellites:
        return {"error": f"sat_id {sat_id} not found"}
    satellite = satellites[sat_id]
    results = {}

    for name, gs in observer_locations.items():
        visible_periods: List[tuple[str, str]] = []
        visible = False
        start: Optional[datetime] = None
        for offset in range(0, 7200, 30):
            future = sim_time + timedelta(seconds=offset)
            t_ts = to_ts(future)
            alt_deg = elevation_deg(satellite, gs, t_ts)
            if alt_deg >= threshold_deg:
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

    return {"sim_time": sim_time.isoformat(), "sat_id": sat_id, "schedule": results}


@app.get("/api/gs/next_comm", tags=["API/GS"])
def get_next_comm_for_sat(sat_id: int = Query(..., description="위성 ID")):
    if sat_id not in satellites:
        return {"error": f"sat_id {sat_id} not found"}
    sat = satellites[sat_id]
    t0 = sim_time
    horizon = 86400
    step = max(1, int(sim_delta_sec))

    for offset in range(step, horizon + step, step):
        t_future = t0 + timedelta(seconds=offset)
        if any(elevation_deg(sat, gs, to_ts(t_future)) >= threshold_deg for gs in observer_locations.values()):
            # 가장 먼저 통신되는 지상국 이름도 함께 찾기
            for name, gs in observer_locations.items():
                if elevation_deg(sat, gs, to_ts(t_future)) >= threshold_deg:
                    return {"sat_id": sat_id, "ground_station": name, "next_comm_time": t_future.isoformat()}
    return {"sat_id": sat_id, "next_comm_time": None, "message": "다음 24시간 내 통신 창 없음"}


@app.get("/api/gs/next_comm_all", tags=["API/GS"])
def get_next_comm_all():
    t0 = sim_time
    horizon = 86400
    step = max(1, int(sim_delta_sec))
    result = {}

    for sat_id, sat in satellites.items():
        next_time = None
        next_gs = None
        for offset in range(step, horizon + step, step):
            t_future = t0 + timedelta(seconds=offset)
            for name, gs in observer_locations.items():
                if elevation_deg(sat, gs, to_ts(t_future)) >= threshold_deg:
                    next_time = t_future
                    next_gs = name
                    break
            if next_time:
                break
        result[sat_id] = {
            "ground_station": next_gs,
            "next_comm_time": next_time.isoformat() if next_time else None
        }
    return {"sim_time": sim_time.isoformat(), "next_comm": result}


@app.put("/api/observer", tags=["API/Observer"])
def set_observer(name: str = Query(...)):
    global observer, current_observer_name
    if name not in observer_locations:
        return {"error": f"observer '{name}' is not supported"}
    observer = observer_locations[name]
    current_observer_name = name
    return {"observer": name, "status": "updated"}


@app.get("/api/observer/check_comm", tags=["API/Observer"])
def get_observer_check_comm(sat_id: int = Query(...)):
    available = bool(sat_comm_status.get(sat_id, False))
    return {"sat_id": sat_id, "observer": current_observer_name, "sim_time": sim_time.isoformat(), "available": available}


@app.get("/api/observer/all_visible", tags=["API/Observer"])
def get_all_visible():
    visible_sats = [sat_id for sat_id, available in sat_comm_status.items() if bool(available)]
    return {"observer": current_observer_name, "sim_time": sim_time.isoformat(), "visible_sat_ids": visible_sats}


@app.get("/api/observer/visible_count", tags=["API/Observer"])
def get_visible_count():
    count = sum(1 for available in sat_comm_status.values() if bool(available))
    return {"observer": current_observer_name, "sim_time": sim_time.isoformat(), "visible_count": count}


@app.get("/api/observer/elevation", tags=["API/Observer"])
def get_elevation(sat_id: int = Query(...)):
    satellite = satellites.get(sat_id)
    if satellite is None:
        return {"error": f"sat_id {sat_id} not found"}
    alt_deg = elevation_deg(satellite, observer, get_current_time_utc())
    return {"sat_id": sat_id, "observer": current_observer_name, "sim_time": sim_time.isoformat(), "elevation_deg": alt_deg}


@app.get("/api/observer/next_comm", tags=["API/Observer"])
def get_next_comm_with_observer(sat_id: int = Query(..., description="위성 ID")):
    if sat_id not in satellites:
        return {"error": f"sat_id {sat_id} not found"}
    sat = satellites[sat_id]
    t0 = sim_time
    horizon = 86400
    step = max(1, int(sim_delta_sec))

    for offset in range(step, horizon + step, step):
        t_future = t0 + timedelta(seconds=offset)
        if elevation_deg(sat, observer, to_ts(t_future)) >= threshold_deg:
            return {"sat_id": sat_id, "observer": current_observer_name, "next_comm_time": t_future.isoformat()}
    return {"sat_id": sat_id, "observer": current_observer_name, "next_comm_time": None, "message": "다음 24시간 내 통신 창 없음"}


@app.get("/api/observer/next_comm_all", tags=["API/Observer"])
def get_next_comm_all_with_observer():
    t0 = sim_time
    horizon = 86400
    step = max(1, int(sim_delta_sec))
    result = {}

    for sat_id, sat in satellites.items():
        next_time = None
        for offset in range(step, horizon + step, step):
            t_future = t0 + timedelta(seconds=offset)
            if elevation_deg(sat, observer, to_ts(t_future)) >= threshold_deg:
                next_time = t_future.isoformat()
                break
        result[sat_id] = {"next_comm_time": next_time}

    return {"observer": current_observer_name, "sim_time": sim_time.isoformat(), "next_comm": result}


@app.get("/api/iot_clusters/position", tags=["API/IoT"])
def get_iot_clusters_position():
    # Topos는 직렬화 불가 → lat/lon만 반환
    clusters = {name: {"lat": t.latitude.degrees, "lon": t.longitude.degrees, "elev_m": raw_iot_clusters[name]["elevation_m"]}
                for name, t in iot_clusters.items()}
    return {"sim_time": sim_time.isoformat(), "clusters": clusters}


@app.get("/api/iot_clusters/visibility_schedule", tags=["API/IoT"])
def get_iot_clusters_visibility(sat_id: int = Query(...)):
    if sat_id not in satellites:
        return {"error": f"sat_id {sat_id} not found"}
    satellite = satellites[sat_id]
    result = []

    for name, cluster in iot_clusters.items():
        visible_periods: List[tuple[str, str]] = []
        visible = False
        start: Optional[datetime] = None
        for offset in range(0, 7200, 30):
            future = sim_time + timedelta(seconds=offset)
            if elevation_deg(satellite, cluster, to_ts(future)) >= threshold_deg:
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
    return {"sim_time": sim_time.isoformat(), "sat_id": sat_id, "schedule": result}


@app.get("/api/iot_clusters/visible", tags=["API/IoT"])
def get_iot_clusters_visible(
    sat_id: Optional[int] = Query(None, description="위성 ID"),
    iot_name: Optional[str] = Query(None, description="IoT 클러스터 이름")
):
    if (sat_id is None) == (iot_name is None):
        return {"error": "neither sat_id or iot_name not found"}

    t_ts = get_current_time_utc()

    if sat_id is not None:
        if sat_id not in satellites:
            return {"error": f"sat_id {sat_id} not found"}
        satellite = satellites[sat_id]
        visible_clusters = [name for name, cluster in iot_clusters.items()
                            if elevation_deg(satellite, cluster, t_ts) >= threshold_deg]
        return {"sat_id": sat_id, "sim_time": sim_time.isoformat(), "visible_iot_clusters": visible_clusters}

    cluster_name = iot_name
    if cluster_name not in iot_clusters:
        return {"error": f"iot cluster '{cluster_name}' not found"}
    cluster = iot_clusters[cluster_name]
    visible_sats = [sid for sid, sat in satellites.items()
                    if elevation_deg(sat, cluster, t_ts) >= threshold_deg]
    return {"sim_time": sim_time.isoformat(), "iot_cluster": cluster_name, "visible_sat_ids": visible_sats}


@app.get("/api/iot_clusters/visible_count", tags=["API/IoT"])
def get_iot_clusters_visible_count(iot_name: str = Query(...)):
    if iot_name not in iot_clusters:
        return {"error": f"iot cluster '{iot_name}' not found"}
    cluster = iot_clusters[iot_name]
    t_ts = get_current_time_utc()
    count = sum(1 for _, sat in satellites.items() if elevation_deg(sat, cluster, t_ts) >= threshold_deg)
    return {"sim_time": sim_time.isoformat(), "iot_cluster": iot_name, "visible_count": count}

@app.get("/api/data/summary", tags=["API/Data"])
def get_data_summary(
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
    sat_ids = sorted(satellites.keys())
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
                    if _has_torch:
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

# ==================== 기타 ====================
async def auto_resume_after_delay():
    global sim_paused, auto_resume_delay_sec
    await asyncio.sleep(auto_resume_delay_sec)
    sim_paused = False
    auto_resume_delay_sec = 0
