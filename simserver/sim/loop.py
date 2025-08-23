# simserver/sim/loop.py
import asyncio
from datetime import timedelta
from .skyfiled import to_ts, elevation_deg
from ..core.logging import make_logger
from ..fl.training import enqueue_training, on_become_visible

logger = make_logger("simserver.loop")

async def run_simulation_loop(ctx):
    while True:
        if not ctx.sim_paused:
            t_ts = to_ts(ctx.sim_time)
            ctx.current_sat_positions = {}
            for sat_id, satellite in ctx.satellites.items():
                prev_visible = bool(ctx.sat_comm_status.get(sat_id, False))
                alt_deg = elevation_deg(satellite, ctx.observer, t_ts)
                visible_now = (alt_deg >= ctx.threshold_deg)
                ctx.sat_comm_status[sat_id] = visible_now

                subpoint = satellite.at(t_ts).subpoint()
                ctx.current_sat_positions[sat_id] = {
                    "lat": subpoint.latitude.degrees,
                    "lon": subpoint.longitude.degrees,
                }

                if not visible_now:
                    enqueue_training(ctx, sat_id)
                elif (not prev_visible) and visible_now:
                    on_become_visible(ctx, sat_id)
                    ctx.train_states[sat_id].round_idx += 1

            ctx.sim_time += timedelta(seconds=ctx.sim_delta_sec)
        await asyncio.sleep(ctx.real_interval_sec)
