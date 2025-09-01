# simserver/sim/state.py
import asyncio, threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Any

from skyfield.api import load, EarthSatellite, Topos
from ..core.logging import make_logger
from ..core.config import Settings
from ..core.hooks import Hooks
from ..core.features import Features
from .skyfiled import to_ts, elevation_deg

logger = make_logger("simserver")

@dataclass
class TrainState:
    running: bool = False
    future: Optional[asyncio.Future] = None
    gpu_id: Optional[int] = None
    stop_event: threading.Event = field(default_factory=threading.Event)
    last_ckpt_path: Optional[str] = None
    round_idx: int = 0
    in_queue: bool = False

@dataclass
class AppState:
    cfg: Settings
    # sim
    sim_time: datetime
    sim_delta_sec: float
    sim_paused: bool
    real_interval_sec: float
    threshold_deg: float
    satellites: Dict[int, EarthSatellite] = field(default_factory=dict)
    sat_comm_status: Dict[int, bool] = field(default_factory=dict)
    current_sat_positions: Dict[int, Dict[str, float]] = field(default_factory=dict)
    observer_locations: Dict[str, Topos] = field(default_factory=dict)
    iot_clusters: Dict[str, Topos] = field(default_factory=dict)
    raw_iot_clusters: Dict[str, Dict[str, float]] = field(default_factory=dict)
    current_observer_name: str = "Berlin"
    observer: Optional[Topos] = None

    # FL globals
    GLOBAL_MODEL_LOCK: threading.Lock = field(default_factory=threading.Lock)
    GLOBAL_MODEL_STATE: Optional[dict] = None
    GLOBAL_MODEL_VERSION: int = -1

    # FL per-sat
    train_states: Dict[int, TrainState] = field(default_factory=dict)

    # caches
    EVAL_DS_CACHE: Dict[str, Any] = field(default_factory=lambda: {"val": None, "test": None})
    EVAL_DONE: set = field(default_factory=set)
    EVAL_DONE_LOCK: threading.Lock = field(default_factory=threading.Lock)

    # executors
    training_executor: Optional[Any] = None
    uploader_executor: Optional[Any] = None

    # queue
    train_queue: Optional[asyncio.Queue] = None

    hooks: Hooks = field(default_factory=Hooks)

    features: Features | None = None

def build_initial_state(cfg: Settings) -> AppState:
    from skyfield.api import Topos

    # observers & iot (원 코드의 상수들 그대로 옮김)
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
        name: Topos(latitude_degrees=cfg2["latitude"], longitude_degrees=cfg2["longitude"], elevation_m=cfg2["elevation_m"])
        for name, cfg2 in raw_iot_clusters.items()
    }

    start = datetime.fromisoformat(cfg.start_time_iso)

    st = AppState(
        cfg=cfg,
        sim_paused=False,
        sim_time=start,
        sim_delta_sec=cfg.sim_delta_sec,
        real_interval_sec=cfg.real_interval_sec,
        threshold_deg=cfg.threshold_deg,
        observer_locations=observer_locations,
        iot_clusters=iot_clusters,
        raw_iot_clusters=raw_iot_clusters,
        current_observer_name="Berlin",
        observer=observer_locations["Berlin"],
    )
    return st

def load_constellation_from_tle(ctx: AppState, tle_path: str):
    if not Path(tle_path).exists():
        raise FileNotFoundError(f"TLE file not found: {tle_path}")

    ts = load.timescale()
    with open(tle_path, "r") as f:
        lines = [line.strip() for line in f.readlines()]
        for i in range(0, len(lines), 3):
            name, line1, line2 = lines[i:i+3]
            sat_id = int(name.replace("SAT", ""))
            ctx.satellites[sat_id] = EarthSatellite(line1, line2, name, ts)

    # initialize positions/status
    t_ts = to_ts(ctx.sim_time)
    for sid, sat in ctx.satellites.items():
        subpoint = sat.at(t_ts).subpoint()
        ctx.current_sat_positions[sid] = {"lat": subpoint.latitude.degrees, "lon": subpoint.longitude.degrees}
        ctx.sat_comm_status[sid] = False
        ctx.train_states[sid] = TrainState()
