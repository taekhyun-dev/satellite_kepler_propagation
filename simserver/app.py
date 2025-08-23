# simserver/app.py
from __future__ import annotations

import os
import asyncio
from contextlib import asynccontextmanager
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
    ctx: AppState = build_initial_state(cfg)
    ctx.features = detect_features()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # === startup ===
        app.state.ctx = ctx
        app.state.tasks = []

        # 1) TLE load
        try:
            tle_path = os.getenv("TLE_PATH", "./constellation.tle")
            load_constellation_from_tle(ctx, tle_path)
        except Exception as e:
            logger.error(f"[INIT] TLE load failed: {e}")
            # 필요시 raise 유지 / 계속 진행할지 결정
            # raise

        # 2) 데이터 레지스트리 초기화
        try:
            from .dataio.registry import (
                DATA_REGISTRY, CIFAR_ROOT, RNG_SEED, DIRICHLET_ALPHA,
                SAMPLES_PER_CLIENT, WITH_REPLACEMENT, ASSIGNMENTS_DIR
            )
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
            ctx.data_ready = True
            ctx.data_error = ""
            logger.info(f"[DATA] CIFAR ready. per_client={SAMPLES_PER_CLIENT}, alpha={DIRICHLET_ALPHA}, seed={RNG_SEED}")
        except Exception as e:
            ctx.data_ready = False
            ctx.data_error = str(e)
            logger.error(f"[DATA] CIFAR init failed: {e}")

        # 3) 글로벌 모델 준비
        try:
            init_global_model(ctx)
        except Exception as e:
            logger.error(f"[AGG] Global model init failed: {e}")
            # 필요시 raise

        # 4) 실행기/워커 시작
        try:
            start_executors_and_workers(ctx)
        except Exception as e:
            logger.error(f"[EXEC] Executors/workers start failed: {e}")

        # 5) 시뮬레이션 루프 시작(백그라운드 태스크로 구동)
        try:
            sim_task = asyncio.create_task(run_simulation_loop(ctx), name="simulation_loop")
            app.state.tasks.append(sim_task)
        except Exception as e:
            logger.error(f"[LOOP] Simulation loop start failed: {e}")

        try:
            yield
        finally:
            # === shutdown ===
            # 백그라운드 태스크 취소
            for t in getattr(app.state, "tasks", []):
                t.cancel()
            if getattr(app.state, "tasks", []):
                await asyncio.gather(*app.state.tasks, return_exceptions=True)

            # 실행기 정리
            try:
                if getattr(ctx, "training_executor", None):
                    ctx.training_executor.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass
            try:
                if getattr(ctx, "uploader_executor", None):
                    ctx.uploader_executor.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass

    app = FastAPI(
        title="Satellite Communication API",
        description="위성·지상국·IoT 통신 상태/가시성/API",
        version="2.0.0",
        lifespan=lifespan,  # ← on_event 대신 lifespan을 등록
    )

    app.include_router(api_router)
    app.include_router(pages_router)

    return app

app = create_app()
