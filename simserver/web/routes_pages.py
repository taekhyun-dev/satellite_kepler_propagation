# simserver/web/routes_pages.py
from __future__ import annotations

import json
from datetime import timedelta
from typing import List, Dict

from fastapi import APIRouter, Query, Request
from fastapi.responses import HTMLResponse

from ..sim.skyfiled import to_ts, elevation_deg

router = APIRouter()

def _ctx(request: Request):
    return request.app.state.ctx  # AppState

# -------------------- Dashboard --------------------
@router.get("/dashboard", response_class=HTMLResponse, tags=["PAGE"])
def dashboard(request: Request):
    ctx = _ctx(request)
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
            .control-form label {{ margin-right: 10px; }}
        </style>
    </head>
    <body>
        <h1>🛰️🛰 Satellite Communication Dashboard</h1>
        <p><b>Sim Time:</b> {ctx.sim_time.isoformat()}</p>
        <p><b>Status:</b> {paused_status}</p>
        <p><b>Step:</b> Δt={ctx.sim_delta_sec}s, Interval={ctx.real_interval_sec}s</p>
        <div class="control-form">
            <label>Δt(초): <input id="delta" type="number" step="any" value="{ctx.sim_delta_sec}" /></label>
            <label>간격(초): <input id="interval" type="number" step="any" value="{ctx.real_interval_sec}" /></label>
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

# -------------------- GS Visibility --------------------
@router.get("/gs_visibility", response_class=HTMLResponse, tags=["PAGE"])
def gs_visibility(request: Request):
    ctx = _ctx(request)
    paused_status = "Running"
    t_ts = to_ts(ctx.sim_time)

    sections = []
    for name, gs in ctx.observer_locations.items():
        rows = []
        for sid, sat in ctx.satellites.items():
            alt_deg = elevation_deg(sat, gs, t_ts)
            if alt_deg >= ctx.threshold_deg:
                rows.append(f'<tr><td>{sid}</td><td>{alt_deg:.2f}°</td></tr>')
        table_html = f"""
        <h2>{name}</h2>
        <table>
            <tr><th>Sat ID</th><th>Elevation</th></tr>
            {''.join(rows)}
        </table>
        """
        sections.append(table_html)

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
        <p><b>Sim Time:</b> {ctx.sim_time.isoformat()}</p>
        <p><b>Status:</b> {paused_status}</p>
        <p><b>Step:</b> Δt={ctx.sim_delta_sec}s, Interval={ctx.real_interval_sec}s</p>
        <hr>
        {''.join(sections)}
    </body>
    </html>
    """

# -------------------- Orbit Path List --------------------
@router.get("/orbit_paths/lists", response_class=HTMLResponse, tags=["PAGE"])
def sat_paths(request: Request):
    ctx = _ctx(request)
    links = [f'<li><a href="/orbit_paths?sat_id={sid}">SAT{sid} Path</a></li>' for sid in sorted(ctx.satellites)]
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

# -------------------- Orbit Path (detail) --------------------
@router.get("/orbit_paths", response_class=HTMLResponse, tags=["PAGE"])
def orbit_paths(request: Request, sat_id: int = Query(...)):
    ctx = _ctx(request)
    if sat_id not in ctx.satellites:
        return HTMLResponse(f"<p>Error: sat_id {sat_id} not found</p>", status_code=404)

    satellite = ctx.satellites[sat_id]
    t0 = ctx.sim_time
    positions: List[tuple[float, float]] = []
    for offset_sec in range(0, 7200, 60):
        future = t0 + timedelta(seconds=offset_sec)
        subpoint = satellite.at(to_ts(future)).subpoint()
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

# -------------------- Map Path (Leaflet) --------------------
@router.get("/map_path", response_class=HTMLResponse, tags=["PAGE"])
def map_path(request: Request):
    ctx = _ctx(request)
    options = ''.join(f'<option value="{sid}">SAT{sid}</option>' for sid in sorted(ctx.satellites))
    obs_data_json = json.dumps({name: {"lat": gs.latitude.degrees, "lon": gs.longitude.degrees}
                                for name, gs in ctx.observer_locations.items()})
    iot_data_json = json.dumps(ctx.raw_iot_clusters)

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
            body {{ font-family: Arial; margin: 1em 1em; }}
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
                center: [0, 0], zoom: 2, worldCopyJump: false,
                maxBounds: [[-85, -180], [85, 180]],
                maxBoundsViscosity: 1.0, inertia: false
            }});
            L.tileLayer('https://{{{{s}}}}.tile.openstreetmap.org/{{{{z}}}}/{{{{x}}}}/{{{{y}}}}.png').addTo(map);

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
                    document.getElementById('sim-step').innerHTML = `<p><b>Step:</b> Δt={ctx.sim_delta_sec}s, Interval={ctx.real_interval_sec}s</p>`;

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

# -------------------- Visibility Schedule List --------------------
@router.get("/visibility_schedules/lists", response_class=HTMLResponse, tags=["PAGE"])
def get_list_visibility_schedules(request: Request):
    ctx = _ctx(request)
    links = [f'<li><a href="/visibility_schedules?sat_id={sid}">SAT{sid} Schedule</a></li>' for sid in sorted(ctx.satellites)]
    return f"""
    <html>
    <head>
        <title>Visibility Schedule List</title>
        <style>
            body {{ font-family: Arial; margin: 2em; }}
            ul {{ list-style-type: none; padding: 0; }}
            li {{ margin-bottom: 10px; }}
        </style>
    </head>
    <body>
        <p><a href="/dashboard">← Back to Dashboard</a></p>
        <h1>📅 All Satellite Visibility Schedules</h1>
        <ul>
            {''.join(links)}
        </ul>
    </body>
    </html>
    """

# -------------------- Visibility Schedule (detail) --------------------
@router.get("/visibility_schedules", response_class=HTMLResponse, tags=["PAGE"])
def visibility_schedules(request: Request, sat_id: int = Query(...)):
    ctx = _ctx(request)
    if sat_id not in ctx.satellites:
        return HTMLResponse(f"<p>Error: sat_id {sat_id} not found</p>", status_code=404)

    satellite = ctx.satellites[sat_id]
    results = []
    for name, gs in ctx.observer_locations.items():
        visible_periods = []
        visible = False
        start = None
        for offset in range(0, 7200, 30):
            future = ctx.sim_time + timedelta(seconds=offset)
            alt_deg = elevation_deg(satellite, gs, to_ts(future))
            if alt_deg >= ctx.threshold_deg:
                if not visible:
                    start = future
                    visible = True
            else:
                if visible:
                    visible_periods.append((start, future))
                    visible = False
        if visible and start:
            visible_periods.append((start, future))
        results.append((name, visible_periods))

    sections = []
    for name, periods in results:
        rows = ''.join(f"<tr><td>{start.strftime('%H:%M:%S')}</td><td>{end.strftime('%H:%M:%S')}</td></tr>" for start, end in periods)
        sections.append(f"<h2>{name}</h2><table><tr><th>Start</th><th>End</th></tr>{rows}</table>")

    return f"""
    <html>
    <head>
        <title>Visibility Schedule</title>
        <meta http-equiv="refresh" content="5">
        <style>
            body {{ font-family: Arial; margin: 2em; }}
            table {{ border-collapse: collapse; width: 60%; margin-bottom: 2em; }}
            th, td {{ border: 1px solid #ccc; padding: 8px; text-align: center; }}
        </style>
    </head>
    <body>
        <p><a href="/visibility_schedules/lists">← Back to Satellite Visibility Schedule List</a></p>
        <h1>📅 Visibility Schedule for SAT{sat_id}</h1>
        <p><b>Sim Time:</b> {ctx.sim_time.isoformat()}</p>
        {''.join(sections)}
    </body>
    </html>
    """

# -------------------- IoT Clusters --------------------
@router.get("/iot_clusters", response_class=HTMLResponse, tags=["PAGE"])
def iot_clusters_ui(request: Request):
    ctx = _ctx(request)
    rows = []
    for name, loc in ctx.raw_iot_clusters.items():
        rows.append(f"<tr><td>{name}</td><td>{loc['latitude']:.2f}</td><td>{loc['longitude']:.2f}</td></tr>")
    raw_json = json.dumps(ctx.raw_iot_clusters)

    return f"""
    <html>
    <head>
        <title>IoT Clusters</title>
        <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
        <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
        <style>
            body {{ font-family: Arial; margin: 2em; }}
            table {{ border-collapse: collapse; width: 60%; }}
            th, td {{ bord
