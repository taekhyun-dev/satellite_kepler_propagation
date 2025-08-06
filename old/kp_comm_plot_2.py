import numpy as np
import datetime
import matplotlib.pyplot as plt
import csv
from PyAstronomy import pyasl

# ======================================================
# 1. Kepler 요소를 직접 이용한 위성 전파 함수들
# ======================================================

def propagate_kepler(ke, t_end, step_seconds=60):
    """
    ke: (a, e, i_deg, RAAN_deg, w_deg, M_deg)
         a: 반장축 (km)
         e: 이심률
         i_deg: 기울기 (deg)
         RAAN_deg: 적경 (deg)
         w_deg: 근지점인자 (deg)
         M_deg: 평균 근점이각 (deg) at epoch (t=0)
    t_end: 전파 시간 (초)
    step_seconds: 시간 해상도 (초)
    
    반환:
      times_sec: 0부터 t_end까지의 시간 (초) 배열
      positions: 각 시간에 대한 위성의 ECI 좌표 (km), shape=(N,3)
    """
    a, e, i_deg, RAAN_deg, w_deg, M_deg = ke
    mu = 398600.4418  # km^3/s^2 (지구)
    # 궤도 주기 (초)
    T = 2 * np.pi * np.sqrt(a**3 / mu)
    n = 2 * np.pi / T  # 평균 운동 (rad/s)
    # epoch에서의 평균 근점이각 M_deg를 rad로 변환하여,
    # t=0에서 M = n*(0 - tau) 이므로, tau = -M/n.
    M_rad = np.deg2rad(M_deg)
    tau = - M_rad / n

    # KeplerEllipse 객체 생성 (각도는 그대로 전달)
    ke_obj = pyasl.KeplerEllipse(a, T, e=e, tau=tau, Omega=RAAN_deg, w=w_deg, i=i_deg)
    times_sec = np.arange(0, t_end + step_seconds, step_seconds)
    positions = np.array([ke_obj.xyzPos(t) for t in times_sec])
    return times_sec, positions

def propagate_kepler_constellation(kepler_elements, epoch_dt, t_end, step_seconds=60):
    """
    kepler_elements: 리스트, 각 원소가 (a, e, i_deg, RAAN_deg, w_deg, M_deg)
    epoch_dt: 전파 시작 시각 (datetime 객체)
    t_end: 전파 총 시간 (초)
    step_seconds: 시간 해상도 (초)
    
    반환:
      constellation_results: 리스트(dict), 각 dict에
         "sat_id": 위성 ID,
         "kepler": 원래 Kepler 요소,
         "times": datetime 객체 리스트 (epoch_dt 기준),
         "coords": ECI 좌표 (km), shape=(N,3)
    """
    constellation_results = []
    for idx, ke in enumerate(kepler_elements):
        times_sec, coords = propagate_kepler(ke, t_end, step_seconds=step_seconds)
        times_dt = [epoch_dt + datetime.timedelta(seconds=float(t)) for t in times_sec]
        sat_info = {"sat_id": idx, "kepler": ke, "times": times_dt, "coords": coords}
        constellation_results.append(sat_info)
    return constellation_results

def generate_walker_delta_constellation(
    total_sat=20,
    num_planes=5,
    altitude_km=570.0,
    inclination_deg=70.0,
    earth_radius_km=6371.0,
    e=0.0,
    arg_perigee_deg=0.0,
    F=1,
    Omega_0=0.0,
    M_0=0.0
):
    """
    Walker Delta 방식으로 Kepler 요소를 생성.
    
    반환:
      kepler_elements: 리스트, 각 원소가 (a, e, i_deg, RAAN_deg, w_deg, M_deg)
    """
    a = earth_radius_km + altitude_km
    sats_per_plane = total_sat // num_planes
    delta_RAAN = 360.0 / num_planes
    delta_M = 360.0 / sats_per_plane
    kepler_elements = []
    for p in range(num_planes):
        for s in range(sats_per_plane):
            RAAN_p = Omega_0 + p * delta_RAAN
            M_ps = (M_0 + s * delta_M + p * (F * 360.0) / (num_planes * sats_per_plane))
            ke = (a, e, inclination_deg, RAAN_p, arg_perigee_deg, M_ps)
            kepler_elements.append(ke)
    return kepler_elements

# ======================================================
# 2. 좌표 변환 및 고도각(elevation) 계산 관련 함수
# ======================================================

def gmst_from_datetime(dt):
    """
    단순 근사식을 이용해 UTC datetime에서 GMST (radians)를 계산.
    J2000.0 기준: 2000-01-01 12:00 UT.
    """
    dt_j2000 = datetime.datetime(2000, 1, 1, 12, 0, 0)
    d = (dt - dt_j2000).total_seconds() / 86400.0
    gmst_hours = (18.697374558 + 24.06570982441908 * d) % 24
    gmst_radians = gmst_hours * (2 * np.pi / 24)
    return gmst_radians

def ecef_from_eci(eci, dt):
    """
    ECI 좌표 (km)를 주어진 시각의 GMST를 이용해 ECEF 좌표로 변환.
    eci: (3,) numpy 배열
    dt: datetime 객체 (UTC)
    """
    gmst = gmst_from_datetime(dt)
    # 회전 행렬: ECEF = R_z(gmst) * ECI
    R = np.array([[ np.cos(gmst),  np.sin(gmst), 0],
                  [-np.sin(gmst),  np.cos(gmst), 0],
                  [           0,             0, 1]])
    return R.dot(eci)

def observer_ecef(lat_deg, lon_deg, elevation_m):
    """
    관측자(지상국)의 ECEF 좌표 (km)를 계산.
    단순 구형 지구 모델: 지구 반지름 6371 km.
    """
    R = 6371.0  # km
    h = elevation_m / 1000.0  # km
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    x = (R + h) * np.cos(lat) * np.cos(lon)
    y = (R + h) * np.cos(lat) * np.sin(lon)
    z = (R + h) * np.sin(lat)
    return np.array([x, y, z])

def topocentric_elevation(sat_ecef, obs_ecef):
    """
    위성의 ECEF 좌표와 관측자의 ECEF 좌표로부터 고도각(elevation, deg)을 계산.
    (관측자 국부 좌표에서 수직 방향은 관측자 위치 벡터 방향으로 가정)
    """
    vec = sat_ecef - obs_ecef
    r = np.linalg.norm(vec)
    obs_unit = obs_ecef / np.linalg.norm(obs_ecef)
    el_rad = np.arcsin(np.dot(vec, obs_unit) / r)
    return np.rad2deg(el_rad)

# ======================================================
# 3. 플롯 및 통신 창(window) 계산 함수 (Kepler 요소 기반)
# ======================================================

def plot_constellation_3d(constellation_results, show_legend=True):
    """
    위성군의 3D ECI 궤적을 플롯.
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    for sat in constellation_results:
        coords = sat["coords"]
        x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
        if show_legend:
            ax.plot(x, y, z, label=f"Sat {sat['sat_id']}")
        else:
            ax.plot(x, y, z)
    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")
    ax.set_zlabel("Z (km)")
    ax.set_title("Walker Delta Constellation 3D Propagation (ECI)")
    if show_legend:
        ax.legend(fontsize=8, loc="upper right")
    plt.show()

def plot_elevation_vs_time_kepler(constellation_results, observer, epoch_dt, title="Elevation vs Time"):
    """
    각 위성에 대해, 관측자(observer, dict: 'lat', 'lon', 'elevation_m') 기준 고도각(elevation)을 시간에 따라 플롯.
    """
    obs_ecef = observer_ecef(observer['lat'], observer['lon'], observer['elevation_m'])
    plt.figure(figsize=(12, 6))
    
    for sat in constellation_results:
        times_dt = sat["times"]
        elevations = []
        for dt, eci in zip(times_dt, sat["coords"]):
            sat_ecef = ecef_from_eci(eci, dt)
            el = topocentric_elevation(sat_ecef, obs_ecef)
            elevations.append(el)
        plt.plot(times_dt, elevations, label=f"Sat {sat['sat_id']}", alpha=0.7)
    
    plt.xlabel("Time (UTC)")
    plt.ylabel("Elevation (°)")
    plt.title(title)
    plt.legend(fontsize=8, loc="upper right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def get_contiguous_windows(times, valid_flags, step_minutes=1):
    """
    Boolean 배열 valid_flags에 대해 연속 구간(창)을 찾아,
    각 구간의 시작, 종료 시간과 지속 시간을 (분) 반환.
    """
    windows = []
    valid = np.array(valid_flags)
    if not np.any(valid):
        return windows
    indices = np.where(valid)[0]
    groups = []
    current_group = [indices[0]]
    for i in indices[1:]:
        if i == current_group[-1] + 1:
            current_group.append(i)
        else:
            groups.append(current_group)
            current_group = [i]
    groups.append(current_group)
    for group in groups:
        start_time = times[group[0]]
        end_time = times[group[-1]]
        duration = (end_time - start_time).total_seconds() / 60.0 + step_minutes
        windows.append((start_time, end_time, duration))
    return windows

def compute_communication_window_details_kepler(constellation_results, observer, threshold_deg=10, step_seconds=60):
    """
    각 위성에 대해, 관측자(observer, dict: 'lat','lon','elevation_m') 기준 고도각이 threshold_deg 이상인 통신 창(window)을 계산.
    
    반환:
      details: 리스트(dict), 각 dict에
         "sat_id": int, "windows": [(start_time, end_time, duration_min), ...],
         "total_minutes": float
    """
    obs_ecef = observer_ecef(observer['lat'], observer['lon'], observer['elevation_m'])
    details = []
    for sat in constellation_results:
        times_dt = sat["times"]
        valid_flags = []
        for dt, eci in zip(times_dt, sat["coords"]):
            sat_ecef = ecef_from_eci(eci, dt)
            el = topocentric_elevation(sat_ecef, obs_ecef)
            valid_flags.append(el >= (90 - threshold_deg))
        windows = get_contiguous_windows(times_dt, valid_flags, step_minutes=step_seconds/60)
        total_minutes = sum([w[2] for w in windows])
        details.append({"sat_id": sat["sat_id"], "windows": windows, "total_minutes": total_minutes})
    return details

def save_comm_windows_csv(comm_details, ground_station_name, filename="communication_windows.csv"):
    """
    통신 창 상세 정보를 CSV 파일로 저장.
    CSV 컬럼: sat_id, ground_station, window_start, window_end, duration_min
    """
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sat_id", "ground_station", "window_start", "window_end", "duration_min"])
        for detail in comm_details:
            sat_id = detail["sat_id"]
            for window in detail["windows"]:
                start_time, end_time, duration = window
                writer.writerow([sat_id, ground_station_name, start_time.isoformat(), end_time.isoformat(), f"{duration:.2f}"])

# ======================================================
# 4. Main 실행: 위성군 전파, 플롯, 통신 창 계산 등
# ======================================================
if __name__ == "__main__":
    # 변수값
    total_sat = 1
    num_planes = 1
    altitude_km = 570.0
    inclination_deg = 70.0
    F = 1
    t_end = 1209600.0  # 초 (예: 2주간 시뮬레이션)
    step_seconds = 60  # 1분 간격

    # 지상국 정보 (dict 형태)
    ground_stations = [
        {'name': 'Houston', 'lat': 29.76, 'lon': -95.37, 'elevation_m': 30},
        {'name': 'Berlin',  'lat': 52.52, 'lon': 13.41,  'elevation_m': 34},
        {'name': 'Tokyo',   'lat': 35.68, 'lon': 139.69, 'elevation_m': 40},
        {'name': 'Nairobi', 'lat': -1.29, 'lon': 36.82,  'elevation_m': 1700},
        {'name': 'Sydney',  'lat': -33.87,'lon': 151.21, 'elevation_m': 58},
    ]
    # Berlin 지상국 선택
    berlin = ground_stations[1]

    # Walker Delta 방식으로 Kepler 요소 생성
    kepler_elements = generate_walker_delta_constellation(
        total_sat=total_sat,
        num_planes=num_planes,
        altitude_km=altitude_km,
        inclination_deg=inclination_deg,
        F=F
    )

    # 전파 시작 시각
    epoch_dt = datetime.datetime(2025, 3, 30, 0, 0, 0)

    # Kepler 요소 기반 위성군 전파
    constellation_results = propagate_kepler_constellation(kepler_elements, epoch_dt, t_end, step_seconds=step_seconds)
    print(f"총 {len(constellation_results)}개 위성 전파 결과 생성됨 (Kepler 요소 기반)")

    # # 3D 플롯 (ECI 좌표)
    # plot_constellation_3d(constellation_results, show_legend=False)

    # # Berlin 지상국 기준 위성 고도각(elevation) vs 시간 플롯
    plot_elevation_vs_time_kepler(constellation_results, berlin, epoch_dt, title="Elevation vs Time from Berlin")

    # # 통신 창 상세 정보 계산 (예: threshold 10° 이상)
    threshold_deg = 10
    comm_details_berlin = compute_communication_window_details_kepler(constellation_results, berlin, threshold_deg=threshold_deg, step_seconds=step_seconds)
    total_comm_berlin = sum([d["total_minutes"] for d in comm_details_berlin])
    print(f"\nBerlin 지상국에서 전체 위성과의 통신 가능 총 시간: {total_comm_berlin:.1f} 분")

    # # CSV 파일로 통신 창 상세 정보 저장 (Berlin)
    save_comm_windows_csv(comm_details_berlin, ground_station_name=berlin['name'], filename="comm_windows_Berlin.csv")
    # print("통신 창 상세 정보 CSV 파일 저장 완료.")
