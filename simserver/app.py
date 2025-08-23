# simserver/app.py
import asyncio
from fastapi import FastAPI
from .core.config import load_settings
from .core.logging import make_logger
from .core.features import detect_features
from .sim.state import build_initial_state, load_constellation_from_tle, AppState
from .fl.aggregate import init_global_model
from .fl.training import start_executors_and_workers
from .sim.loop import run_simulation_loop
from .web.routes_api import router as api_router
from .web.routes_pages import router as pages_router

logger = make_logger("simserver")

def create_app() -> FastAPI:
    cfg = load_settings()
    ctx = build_initial_state(cfg)
    ctx.features = detect_features()

    app = FastAPI(
        title="Satellite Communication API",
        description="위성·지상국·IoT 통신 상태/가시성/API",
        version="2.0.0"
    )
    app.include_router(api_router)
    app.include_router(pages_router)
    app.state.ctx = ctx

    @app.on_event("startup")
    async def _startup():
        # TLE load
        tle_path = "./constellation.tle"  # 필요시 환경변수로 변경
        load_constellation_from_tle(ctx, tle_path)

        # 데이터 레지스트리 초기화 (원 코드 그대로)
        try:
            from .dataio.registry import DATA_REGISTRY, CIFAR_ROOT, RNG_SEED, DIRICHLET_ALPHA, SAMPLES_PER_CLIENT, WITH_REPLACEMENT, ASSIGNMENTS_DIR
            DATA_REGISTRY.load_base(root=CIFAR_ROOT, seed=RNG_SEED, download=True)
            assign_file = ASSIGNMENTS_DIR / f"assign_seed{RNG_SEED}_alpha{DIRICHLET_ALPHA}_spc{SAMPLES_PER_CLIENT}.npz"
            if not DATA_REGISTRY.load_assignments(assign_file):
                sat_ids_sorted = sorted(ctx.satellites.keys())
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

        # 글로벌 모델 준비
        init_global_model(ctx)

        # 실행기/워커 시작
        start_executors_and_workers(ctx)

        # 시뮬레이션 루프 시작
        asyncio.create_task(run_simulation_loop(ctx))

    @app.on_event("shutdown")
    async def _shutdown():
        try:
            ctx.training_executor.shutdown(wait=False, cancel_futures=True)
        except Exception: pass
    return app

app = create_app()
