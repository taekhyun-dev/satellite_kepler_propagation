# sat_comm_server.py
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse
from fastapi import Request
from contextlib import asynccontextmanager
from skyfield.api import load, EarthSatellite, Topos
from datetime import datetime, timedelta
from typing import Optional
import asyncio
import os

# ==================== 시뮬레이션 상태 변수 ====================
satellites = {}           # sat_id -> EarthSatellite 객체
sat_comm_status = {}      # sat_id -> 현재 통신 가능 여부
observer_locations = {    # observer 이름 -> Topos 객체
    "Berlin": Topos(latitude_degrees=52.52, longitude_degrees=13.41, elevation_m=34),
    "Houston": Topos(latitude_degrees=29.76, longitude_degrees=-95.37, elevation_m=30),
    "Tokyo": Topos(latitude_degrees=35.68, longitude_degrees=139.69, elevation_m=40),
    "Nairobi": Topos(latitude_degrees=-1.29, longitude_degrees=36.82, elevation_m=1700),
    "Sydney": Topos(latitude_degrees=-33.87, longitude_degrees=151.21, elevation_m=58)
}

raw_iot_clusters = {
    "Abisko": {"latitude": 68.35, "longitude": 18.79, "elevation_m": 420},
    "Boreal": {"latitude": 55.50, "longitude": 105.00, "elevation_m": 450},
    "Taiga": {"latitude": 58.00, "longitude": 99.00, "elevation_m": 300},
    "Patagonia": {"latitude": 51.00, "longitude": 73.00, "elevation_m": 500},
    "Amazon_Forest": {"latitude": -3.47, "longitude": -62.37, "elevation_m": 100},  # 아마존 열대우림
    "Great_Barrier": {"latitude": -18.29, "longitude": 147.77, "elevation_m": 0},   # 그레이트 배리어 리프
    "Mediterranean": {"latitude": 37.98, "longitude": 23.73, "elevation_m": 170},    # 지중해 연안
    "California": {"latitude": 36.78, "longitude": -119.42, "elevation_m": 150}
}

iot_clusters = {
    "Abisko": Topos(latitude_degrees= 68.35, longitude_degrees= 18.79, elevation_m=420),
    "Boreal": Topos(latitude_degrees= 55.50, longitude_degrees= 105.00, elevation_m=450),
    "Taiga": Topos(latitude_degrees= 58.00, longitude_degrees= 99.00, elevation_m=300),
    "Patagonia": Topos(latitude_degrees= 51.00, longitude_degrees= 73.00, elevation_m=500),
    "Amazon_Forest": Topos(latitude_degrees= -3.47, longitude_degrees= -62.37, elevation_m=100),  # 아마존 열대우림
    "Great_Barrier": Topos(latitude_degrees= -18.29, longitude_degrees= 147.77, elevation_m=0),   # 그레이트 배리어 리프
    "Mediterranean": Topos(latitude_degrees= 37.98, longitude_degrees= 23.73, elevation_m=170),    # 지중해 연안
    "California": Topos(latitude_degrees= 36.78, longitude_degrees= -119.42, elevation_m=150),
}
current_observer_name = "Berlin"
observer = observer_locations[current_observer_name]
ts = load.timescale()
sim_time = datetime(2025, 3, 30, 0, 0, 0)  # 시뮬레이션 시작 시간
threshold_deg = 40
sim_speed = 20.0  # 20배속 (0.05초에 1초 시뮬레이션)
sim_paused = False  # 시뮬레이션 일시정지 여부
auto_resume_delay_sec = 0  # 자동 재개 지연 시간 (초)
current_sat_positions = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    await initialize_simulation()
    yield

app = FastAPI(
    title="Satellite Communication API",
    description="위성, 지상국, IoT 클러스터 간 통신 상태 및 관측 가능 시간 등을 제공하는 API 서비스입니다.",
    version="1.0.0",
    docs_url="/docs",       # Swagger UI
    redoc_url="/redoc",     # ReDoc UI
    openapi_url="/openapi.json",  # OpenAPI 스키마 URL
    lifespan=lifespan
)

# ==================== 서버 기동 시 처리 ====================
async def initialize_simulation():
    global satellites

    tle_path = "../constellation.tle"
    if not os.path.exists(tle_path):
        raise FileNotFoundError("TLE 파일이 존재하지 않습니다: constellation.tle")

    with open(tle_path, "r") as f:
        lines = [line.strip() for line in f.readlines()]
        for i in range(0, len(lines), 3):
            name, line1, line2 = lines[i:i+3]
            sat_id = int(name.replace("SAT", ""))
            satellite = EarthSatellite(line1, line2, name, ts)
            satellites[sat_id] = satellite
            
    # 초기 위치 계산 추가
    t = ts.utc(sim_time.year, sim_time.month, sim_time.day,
               sim_time.hour, sim_time.minute, sim_time.second)
    for sat_id, satellite in satellites.items():
        subpoint = satellite.at(t).subpoint()
        current_sat_positions[sat_id] = {
            "lat": subpoint.latitude.degrees,
            "lon": subpoint.longitude.degrees
        }

    asyncio.create_task(simulation_loop())

# ==================== 시뮬레이션 루프 ====================
async def simulation_loop():
    global sim_time, current_sat_positions
    while True:
        if not sim_paused:
            t = ts.utc(sim_time.year, sim_time.month, sim_time.day,
                    sim_time.hour, sim_time.minute, sim_time.second)
            current_sat_positions = {}
            for sat_id, satellite in satellites.items():
                difference = satellite - observer
                topocentric = difference.at(t)
                alt, az, dist = topocentric.altaz()
                sat_comm_status[sat_id] = alt.degrees >= threshold_deg
                subpoint = satellite.at(t).subpoint()
                current_sat_positions[sat_id] = {
                    "lat": subpoint.latitude.degrees,
                    "lon": subpoint.longitude.degrees
                }

            sim_time += timedelta(seconds=1)      # 시뮬레이션 시간 증가
        await asyncio.sleep(1.0 / sim_speed)             # 빠르게 진행 (20배속)

# ==================== 대시보드 HTML UI ====================
@app.get("/dashboard", response_class=HTMLResponse, tags=["PAGE"])
def dashboard():
    """
    대시보드 HTML 페이지
    """
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
        <p><b>Speed:</b> {sim_speed}x</p>
        <hr>
        <a href="/gs_visibility">🛰️ GS별 통신 가능 위성 보기</a>
        <a href="/orbit_paths/lists">🛰 위성별 궤적 경로 보기</a>
        <a href="/map_path">🗺 지도 기반 위성 경로 보기</a>
        <a href="/visibility_schedules/lists">📅 위성별 관측 가능 시간대 목록 보기</a>
        <a href="/iot_clusters"> 📡 IoT 클러스터별 위치 보기</a>
        <a href="/iot_visibility"> 🌐 IoT 클러스터별 통신 가능 위성 보기</a>
    </body>
    </html>
    """


@app.get("/gs_visibility", response_class=HTMLResponse, tags=["PAGE"])
def gs_visibility():
    """
    지상국별로 관측 가능한 위성 목록을 HTML로 반환하는 페이지
    """
    gs_sections = []
    for name, gs in observer_locations.items():
        t = ts.utc(sim_time.year, sim_time.month, sim_time.day, sim_time.hour, sim_time.minute, sim_time.second)
        rows = []
        for sid, sat in satellites.items():
            difference = sat - gs
            topocentric = difference.at(t)
            alt, az, dist = topocentric.altaz()
            if alt.degrees >= threshold_deg:
                rows.append(f'<tr><td>{sid}</td><td>{alt.degrees:.2f}°</td></tr>')
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
        <hr>
        {''.join(gs_sections)}
    </body>
    </html>
    """

@app.get("/orbit_paths/lists", response_class=HTMLResponse, tags=["PAGE"])
def sat_paths():
    """
    위성별 궤적 경로 링크 목록을 HTML로 반환하는 페이지
    """
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
    """
    특정 위성의 궤적 경로를 HTML로 반환하는 페이지
    """
    if sat_id not in satellites:
        return HTMLResponse(f"<p>Error: sat_id {sat_id} not found</p>", status_code=404)

    satellite = satellites[sat_id]
    t0 = sim_time
    positions = []
    for offset_sec in range(0, 7200, 60):
        future = t0 + timedelta(seconds=offset_sec)
        t = ts.utc(future.year, future.month, future.day, future.hour, future.minute, future.second)
        subpoint = satellite.at(t).subpoint()
        lat = subpoint.latitude.degrees
        lon = subpoint.longitude.degrees
        positions.append((lat, lon))

    rows = ''.join([f'<tr><td>{i*60}s</td><td>{lat:.2f}</td><td>{lon:.2f}</td></tr>' for i, (lat, lon) in enumerate(positions)])

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
        <p><a href="/sat_paths">← Back to All Satellite Orbit Paths</a></p>
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
    """
    지도 기반 위성 경로를 표시하는 HTML 페이지
    """
    t0 = sim_time
    options = ''.join(f'<option value="{sid}">SAT{sid}</option>' for sid in sorted(satellites))
    return f"""
    <html>
    <head>
        <title>Map Path</title>
        <link rel=\"stylesheet\" href=\"https://unpkg.com/leaflet@1.9.4/dist/leaflet.css\" />
        <script src=\"https://unpkg.com/leaflet@1.9.4/dist/leaflet.js\"></script>
        <style>
            #map {{ height: 90vh; }}
            html, body {{ margin: 0; padding: 0; }}
        </style>
    </head>
    <body>
        <p><a href=\"/dashboard\">← Back to Dashboard</a></p>
        <h1>🗺 Satellite Map Path</h1>
        <div id="sim-time"></div>
        <label for=\"sat_id\">Choose a satellite:</label>
        <select id=\"sat_id\" onchange=\"drawTrajectory(this.value)\">{options}</select>
        <div id=\"map\"></div>
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
            var circles = [];
            var pathLines = [];
            var currentMarker = null;
            var currentLabel = null;
            var markerInterval = null;

            async function drawTrajectory(sat_id) {{
                circles.forEach(c => map.removeLayer(c));
                circles = [];
                pathLines.forEach(p => map.removeLayer(p));
                pathLines = [];
                if (currentMarker) map.removeLayer(currentMarker);
                if (currentLabel) map.removeLayer(currentLabel);
                if (markerInterval) clearInterval(markerInterval);

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
                    document.getElementById('sim-time').innerText = `현재 시뮬레이션 시간: ${{simData.sim_time}}`;

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

            // 초기 로딩 시 첫 위성 표시
            window.onload = () => {{
                const selector = document.getElementById('sat_id');
                if (selector.value) drawTrajectory(selector.value);
            }}
        </script>
    </body>
    </html>
    """

@app.get("/visibility_schedules/lists", response_class=HTMLResponse, tags=["PAGE"])
def get_list_visibility_schedules():
    """
    위성별 관측 가능 시간대 링크 목록을 HTML로 반환하는 페이지
    """
    links = [f'<li><a href="/visibility_schedules?sat_id={sid}">SAT{sid} Schedule</a></li>' for sid in sorted(satellites)]
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

@app.get("/visibility_schedules", response_class=HTMLResponse, tags=["PAGE"])
def visibility_schedules(sat_id: int = Query(...)):
    """
    특정 위성의 관측 가능 시간대를 HTML로 반환하는 페이지
    """
    if sat_id not in satellites:
        return HTMLResponse(f"<p>Error: sat_id {sat_id} not found</p>", status_code=404)

    satellite = satellites[sat_id]
    results = []
    for name, gs in observer_locations.items():
        visible_periods = []
        visible = False
        start = None
        for offset in range(0, 7200, 30):
            future = sim_time + timedelta(seconds=offset)
            t = ts.utc(future.year, future.month, future.day, future.hour, future.minute, future.second)
            difference = satellite - gs
            topocentric = difference.at(t)
            alt, _, _ = topocentric.altaz()
            if alt.degrees >= threshold_deg:
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
        <p><b>Sim Time:</b> {sim_time.isoformat()}</p>
        {''.join(sections)}
    </body>
    </html>
    """

@app.get("/iot_clusters", response_class=HTMLResponse, tags=["PAGE"])
def iot_clusters_ui():
    """
    IoT 클러스터 위치를 HTML로 반환하는 페이지
    """
    rows = []
    for name, loc in raw_iot_clusters.items():
        # console.log(f"Adding IoT cluster: {name} at {loc['latitude']}, {loc['longitude']}")
        rows.append(f"<tr><td>{name}</td><td>{loc['latitude']:.2f}</td><td>{loc['longitude']:.2f}</td></tr>")
    return f"""
    <html>
    <head>
        <title>IoT Clusters</title>
        <link rel=\"stylesheet\" href=\"https://unpkg.com/leaflet@1.9.4/dist/leaflet.css\" />
        <script src=\"https://unpkg.com/leaflet@1.9.4/dist/leaflet.js\"></script>
        <style>
            body {{ font-family: Arial; margin: 2em; }}
            table {{ border-collapse: collapse; width: 60%; }}
            th, td {{ border: 1px solid #ccc; padding: 8px; text-align: center; }}
            #map {{ height: 500px; margin-top: 30px; }}
        </style>
    </head>
    <body>
        <p><a href="/dashboard">← Back to Dashboard</a></p>
        <h1>📡 IoT Cluster Locations</h1>
        <table>
            <tr><th>Name</th><th>Latitude</th><th>Longitude</th></tr>
            {''.join(rows)}
        </table>
        <hr/>
        <div id=\"map\"></div>
        <script>
            var map = L.map('map', {{
                center: [0, 0],
                zoom: 2,
                worldCopyJump: false,
                maxBounds: [[-85, -180], [85, 180]],
                maxBoundsViscosity: 1.0,
                inertia: false
            }});
            L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
                maxZoom: 18,
                attribution: '© OpenStreetMap contributors'
            }}).addTo(map);

            const clusters = {raw_iot_clusters};
            console.log("IoT clusters:", clusters);
            for (let [name, loc] of Object.entries(clusters)) {{
                const lat = loc.latitude;
                const lon = loc.longitude;
                L.circleMarker([lat, lon],{{radius: 5, color: 'blue'}}).addTo(map);
                const marker = L.circleMarker([lat, lon], {{ radius: 5, color: 'blue' }}).addTo(map);
                marker.bindTooltip(name, {{
                    permanent: true,
                    direction: 'top',
                    className: 'iot-tooltip'
                }});
            }}
        </script>
    </body>
    </html>
    """

@app.get("/iot_visibility", response_class=HTMLResponse, tags=["PAGE"])
def iot_visibility():
    """
    IoT 클러스터에서 관측 가능한 위성 목록을 HTML로 반환하는 페이지
    """
    iot_sections = []
    for name, iot in iot_clusters.items():
        t = ts.utc(sim_time.year, sim_time.month, sim_time.day, sim_time.hour, sim_time.minute, sim_time.second)
        rows = []
        for sid, sat in satellites.items():
            difference = sat - iot
            topocentric = difference.at(t)
            alt, az, dist = topocentric.altaz()
            if alt.degrees >= threshold_deg:
                rows.append(f'<tr><td>{sid}</td><td>{alt.degrees:.2f}°</td></tr>')
        table_html = f"""
        <h2>{name}</h2>
        <table>
            <tr><th>Sat ID</th><th>Elevation</th></tr>
            {''.join(rows)}
        </table>
        """
        iot_sections.append(table_html)

    return f"""
    <html>
    <head>
        <title>IOT Visibility</title>
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
        <h1>🌐 IOT-wise Visible Satellites</h1>
        <p><b>Sim Time:</b> {sim_time.isoformat()}</p>
        <hr>
        {''.join(iot_sections)}
    </body>
    </html>
    """

# ==================== 클라이언트 요청 ====================
@app.post("/api/reset_time", tags=["API"])
def reset_sim_time():
    """
    시뮬레이션 시간을 초기화하는 API
    """
    global sim_time
    sim_time = datetime(2025, 3, 30, 0, 0, 0)
    return {"status": "reset", "sim_time": sim_time.isoformat()}

@app.get("/api/sim_time", tags=["API"])
def get_sim_time():
    """
    현재 시뮬레이션 시간을 반환하는 API
    """
    return {"sim_time": sim_time.isoformat()}

@app.put("/api/sim_time", tags=["API"])
def set_sim_time(year: int, month: int, day: int, hour: int = 0, minute: int = 0, second: int = 0):
    """
    시뮬레이션 시간을 설정하는 API
    """
    global sim_time
    try:
        sim_time = datetime(year, month, day, hour, minute, second)
        return {"status": "updated", "sim_time": sim_time.isoformat()}
    except Exception as e:
        return {"error": str(e)}

@app.put("/api/set_speed", tags=["API"])
def set_sim_speed(speed: float):
    """
    시뮬레이션 속도(배속)를 설정하는 API
    """
    global sim_speed
    if speed <= 0:
        return {"error": "speed must be positive"}
    sim_speed = speed
    return {"status": "updated", "sim_speed": sim_speed}

@app.post("/api/pause", tags=["API"])
def pause_simulation():
    """
    시뮬레이션을 일시정지하는 API
    """
    global sim_paused
    sim_paused = True
    return {"status": "paused"}

@app.post("/api/resume", tags=["API"])
def resume_simulation():
    """
    시뮬레이션을 재개하는 API
    """
    global sim_paused
    sim_paused = False
    return {"status": "resumed"}

@app.get("/api/trajectory", tags=["API"])
def get_trajectory(sat_id: int = Query(...)):
    """
    특정 위성의 궤적 경로를 반환하는 API
    """
    if sat_id not in satellites:
        return {"error": f"sat_id {sat_id} not found"}
    satellite = satellites[sat_id]
    t0 = sim_time
    prev_lon = None
    segment = []
    segments = []

    for offset_sec in range(0, 7200, 30):
        future = t0 + timedelta(seconds=offset_sec)
        t = ts.utc(future.year, future.month, future.day, future.hour, future.minute, future.second)
        subpoint = satellite.at(t).subpoint()
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
    """
    특정 위성의 현재 위치를 반환하는 API
    """
    if sat_id not in current_sat_positions:
        return {"error": f"Position for SAT{sat_id} not available"}
    return current_sat_positions[sat_id]

@app.get("/api/gs/check_comm", tags=["API/GS"])
def get_gs_check_comm(sat_id: int = Query(...)):
    """
    위성 통신 가능 여부를 확인하는 API
    """
    available = bool(sat_comm_status.get(sat_id, False))
    return {
        "sat_id": sat_id,
        "observer": "Berlin",
        "sim_time": sim_time.isoformat(),
        "available": available
    }

@app.put("/api/gs/observer", tags=["API/GS"])
def set_observer(name: str = Query(...)):
    """
    지상국 관측 위치를 설정하는 API
    """
    global observer, current_observer_name
    if name not in observer_locations:
        return {"error": f"observer '{name}' is not supported"}
    observer = observer_locations[name]
    current_observer_name = name
    return {"observer": name, "status": "updated"}

@app.get("/api/gs/visibility", tags=["API/GS"])
def get_gs_visibility():
    """
    현재 시뮬레이션 시간에 각 지상국에서 관측 가능한 위성 목록을 반환하는 API
    """
    result = {}
    t = ts.utc(sim_time.year, sim_time.month, sim_time.day,
               sim_time.hour, sim_time.minute, sim_time.second)
    for name, gs in observer_locations.items():
        visible_sats = []
        for sid, sat in satellites.items():
            difference = sat - gs
            topocentric = difference.at(t)
            alt, _, _ = topocentric.altaz()
            if alt.degrees >= threshold_deg:
                visible_sats.append({"sat_id": sid, "elevation": alt.degrees})
        result[name] = visible_sats
    return {"sim_time": sim_time.isoformat(), "data": result}

@app.get("/api/gs/visibility_schedule", tags=["API/GS"])
def get_visibility_schedule(sat_id: int = Query(...)):
    """
    특정 위성의 관측 가능 시간대를 반환하는 API
    """
    if sat_id not in satellites:
        return {"error": f"sat_id {sat_id} not found"}
    satellite = satellites[sat_id]
    results = {}
    for name, gs in observer_locations.items():
        visible_periods = []
        visible = False
        start = None
        for offset in range(0, 7200, 30):
            future = sim_time + timedelta(seconds=offset)
            t = ts.utc(future.year, future.month, future.day, future.hour, future.minute, future.second)
            difference = satellite - gs
            topocentric = difference.at(t)
            alt, _, _ = topocentric.altaz()
            if alt.degrees >= threshold_deg:
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

@app.get("/api/gs/all_visible", tags=["API/GS"])
def get_all_visible():
    """
    현재 시뮬레이션 시간에 지상국에서 관측 가능한 모든 위성의 ID를 반환하는 API
    """
    visible_sats = [sat_id for sat_id, available in sat_comm_status.items() if bool(available)]
    return {
        "observer": current_observer_name,
        "sim_time": sim_time.isoformat(),
        "visible_sat_ids": visible_sats
    }

@app.get("/api/gs/visible_count", tags=["API/GS"])
def get_visible_count():
    """
    현재 시뮬레이션 시간에 지상국에서 관측 가능한 위성의 개수를 반환하는 API
    """
    count = sum(1 for available in sat_comm_status.values() if bool(available))
    return {
        "observer": current_observer_name,
        "sim_time": sim_time.isoformat(),
        "visible_count": count
    }

@app.get("/api/gs/elevation", tags=["API/GS"])
def get_elevation(sat_id: int = Query(...)):
    """
    특정 위성의 현재 시뮬레이션 시간에 대한 지상국과의 고도를 반환하는 API
    """
    satellite = satellites.get(sat_id)
    if satellite is None:
        return {"error": f"sat_id {sat_id} not found"}
    t = ts.utc(sim_time.year, sim_time.month, sim_time.day,
               sim_time.hour, sim_time.minute, sim_time.second)
    difference = satellite - observer
    topocentric = difference.at(t)
    alt, az, dist = topocentric.altaz()
    return {
        "sat_id": sat_id,
        "observer": current_observer_name,
        "sim_time": sim_time.isoformat(),
        "elevation_deg": alt.degrees
    }

@app.get("/api/iot_clusters/position", tags=["API/IoT"])
def get_iot_clusters_position():
    """
    IoT 클러스터의 위치 정보를 반환하는 API
    """
    return {"sim_time": sim_time.isoformat(), "clusters": iot_clusters}

@app.get("/api/iot_clusters/visibility_schedule", tags=["API/IoT"])
def get_iot_clusters_visibility(sat_id: int = Query(...)):
    """
    특정 위성이 IoT 클러스터에서 관측 가능한지 여부와 관측 가능 시간대를 반환하는 API
    """
    if sat_id not in satellites:
        return {"error": f"sat_id {sat_id} not found"}

    satellite = satellites[sat_id]
    result = []

    for name, cluster in iot_clusters.items():
        visible_periods = []
        visible = False
        start = None
        for offset in range(0, 7200, 30):
            future = sim_time + timedelta(seconds=offset)
            t = ts.utc(future.year, future.month, future.day, future.hour, future.minute, future.second)
            difference = satellite - cluster
            topocentric = difference.at(t)
            alt, _, _ = topocentric.altaz()
            if alt.degrees >= threshold_deg:
                if not visible:
                    start = future
                    visible = True
            else:
                if visible:
                    visible_periods.append((start, future))
                    visible = False
        if visible and start:
            visible_periods.append((start, future))
  
        if visible_periods:
            result.append({"iot_cluster": name, "periods": [(s.isoformat(), e.isoformat()) for s, e in visible_periods]})

    return {"sim_time": sim_time.isoformat(), "sat_id": sat_id, "schedule": result}

@app.get("/api/iot_clusters/visible", tags=["API/IoT"])
def get_iot_clusters_visible(
        sat_id: Optional[int] = Query(None, description="위성 ID"),
        iot_name: Optional[str] = Query(None, description="IoT 클러스터 이름")
    ):
    """
    특정 IoT 클러스터에서 관측 가능한지 여부를 반환하는 API\n
    - sat_id만 지정 → 해당 위성과 통신 가능한 IoT 클러스터 목록 반환
    - iot_name만 지정 → 해당 IoT 클러스터와 통신 가능한 위성 ID 목록 반환
    """
    if (sat_id is None) == (iot_name is None):
        return {"error": "neither sat_id or iot_name not found"}
    t = ts.utc(sim_time.year, sim_time.month, sim_time.day, sim_time.hour, sim_time.minute, sim_time.second)

    if sat_id is not None:
        if sat_id not in satellites:
            return {"error": f"sat_id {sat_id} not found"}
        
        satellite = satellites[sat_id]
        visible_clusters = []

        for name, cluster in iot_clusters.items():
            difference = satellite - cluster
            topocentric = difference.at(t)
            alt, _, _ = topocentric.altaz()
            if alt.degrees >= threshold_deg:
                visible_clusters.append(name)

        return {
            "sat_id": sat_id,
            "sim_time": sim_time.isoformat(),
            "visible_iot_clusters": visible_clusters
        }
    
    cluster_name = iot_name
    if cluster_name not in iot_clusters:
        return {"error": f"iot cluster '{cluster_name}' not found"}
    cluster = iot_clusters[cluster_name]
    visible_sats = []
    for sid, sat in satellites.items():
        alt, _, _ = (sat - cluster).at(t).altaz()
        if alt.degrees >= threshold_deg:
            visible_sats.append(sid)
    return {
        "sim_time": sim_time.isoformat(),
        "iot_cluster": cluster_name,
        "visible_sat_ids": visible_sats
    }

@app.get("/api/iot_clusters/visible_count", tags=["API/IoT"])
def get_iot_clusters_visible_count(iot_name: str = Query(...)):
    """
    특정 IoT 클러스터에서 관측 가능한 위성의 개수를 반환하는 API
    """
    if iot_name not in iot_clusters:
        return {"error": f"iot cluster '{iot_name}' not found"}

    cluster = iot_clusters[iot_name]
    count = 0
    t = ts.utc(sim_time.year, sim_time.month, sim_time.day, sim_time.hour, sim_time.minute, sim_time.second)

    for sat_id, sat in satellites.items():
        difference = sat - cluster
        topocentric = difference.at(t)
        alt, _, _ = topocentric.altaz()
        if alt.degrees >= threshold_deg:
            count += 1

    return {
        "sim_time": sim_time.isoformat(),
        "iot_cluster": iot_name,
        "visible_count": count
    }

async def auto_resume_after_delay():
    global sim_paused, auto_resume_delay_sec
    await asyncio.sleep(auto_resume_delay_sec)
    sim_paused = False
    auto_resume_delay_sec = 0
