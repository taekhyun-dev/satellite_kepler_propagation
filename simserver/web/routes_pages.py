# simserver/web/routes_pages.py
from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter()

# 아래 3개 페이지만 예시로 옮겼어. (나머지도 동일 패턴)
@router.get("/dashboard", response_class=HTMLResponse, tags=["PAGE"])
def dashboard(request):
    """
    대시보드 HTML 페이지
    """
    ctx = request.app.state.ctx
    paused_status = "Paused" if ctx.sim_paused else "Running"
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
        <p><b>Sim Time:</b> {ctx.sim_time.isoformat()}</p>
        <p><b>Status:</b> {paused_status}</p>
        <p><b>Step:</b> Δt={ctx.sim_delta_sec}s, Interval={ctx.real_interval_sec}s</p>
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
        <a href="/gs_visibility">🛰️ GS별 통신 가능 위성 보기</a><br/>
        <a href="/orbit_paths/lists">🛰 위성별 궤적 경로 보기</a><br/>
        <a href="/map_path">🗺 지도 기반 위성 경로 보기</a><br/>
        <a href="/visibility_schedules/lists">📅 위성별 관측 가능 시간대 목록 보기</a><br/>
        <a href="/iot_clusters"> 📡 IoT 클러스터별 위치 보기</a><br/>
        <a href="/iot_visibility"> 🌐 IoT 클러스터별 통신 가능 위성 보기</a><br/>
        <a href="/comm_targets/lists">🚀 위성 통신 대상 확인</a>
    </body>
    </html>
    """

@router.get("/gs_visibility", response_class=HTMLResponse, tags=["PAGE"])
def gs_visibility(request):
    """
    지상국별로 관측 가능한 위성 목록을 HTML로 반환하는 페이지
    """
    ctx = request.app.state.ctx
    t_ts = __import__("simserver.sim.sky", fromlist=[""]).to_ts(ctx.sim_time)
    rows_by_gs = []
    for name, gs in ctx.observer_locations.items():
        rows=[]
        for sid, sat in ctx.satellites.items():
            alt_deg = __import__("simserver.sim.sky", fromlist=[""]).elevation_deg(sat, gs, t_ts)
            if alt_deg >= ctx.threshold_deg:
                rows.append(f'<tr><td>{sid}</td><td>{alt_deg:.2f}°</td></tr>')
        rows_by_gs.append(f"<h2>{name}</h2><table><tr><th>Sat ID</th><th>Elevation</th></tr>{''.join(rows)}</table>")
    return HTMLResponse(f"<html><body><p><a href='/dashboard'>← Back</a></p><h1>GS Visibility</h1>{''.join(rows_by_gs)}</body></html>")

@router.get("/orbit_paths/lists", response_class=HTMLResponse, tags=["PAGE"])
def sat_paths(request):
    """
    위성별 궤적 경로 링크 목록을 HTML로 반환하는 페이지
    """
    ctx = request.app.state.ctx
    links = [f'<li><a href="/orbit_paths?sat_id={sid}">SAT{sid} Path</a></li>' for sid in sorted(ctx.satellites)]
    return HTMLResponse(f"<html><body><p><a href='/dashboard'>← Back</a></p><ul>{''.join(links)}</ul></body></html>")

@app.get("/orbit_paths", response_class=HTMLResponse, tags=["PAGE"])
def orbit_paths(request, sat_id: int = Query(...)):
    """
    특정 위성의 궤적 경로를 HTML로 반환하는 페이지
    """
    ctx = request.app.state.ctx
    if sat_id not in ctx.satellites:
        return HTMLResponse(f"<p>Error: sat_id {sat_id} not found</p>", status_code=404)

    satellite = ctx.satellites[sat_id]
    t0 = ctx.sim_time
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