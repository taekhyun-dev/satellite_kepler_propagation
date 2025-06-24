import numpy as np
import datetime
from sgp4.api import Satrec, WGS72, jday
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import csv
from PyAstronomy import pyasl
from skyfield.api import load, EarthSatellite, Topos
import math

# ----------------------------------------------------
# TLE 체크섬 계산 함수
# ----------------------------------------------------
def compute_tle_checksum(line: str) -> int:
    """
    TLE 행(최대 68자)에 대한 체크섬을 계산합니다.
    숫자는 그대로 더하고, '-' 문자는 1로 취급하며,
    그 외 문자는 0으로 취급하여, 최종 합 mod 10을 반환합니다.
    """
    checksum = 0
    for char in line[:68]:
        if char.isdigit():
            checksum += int(char)
        elif char == '-':
            checksum += 1
    return checksum % 10

# ----------------------------------------------------
# 기존 TLE 기반 위성군 전파 관련 함수들
# ----------------------------------------------------
def propagate_leo(a, per, e, tau, Omega, w, i, num_points=200):
    ke = pyasl.KeplerEllipse(a, per, e=e, tau=tau, Omega=Omega, w=w, i=i)
    times = np.linspace(0, per, num_points)
    coords = np.zeros((num_points, 3), dtype=float)
    for idx, t in enumerate(times):
        x, y, z = ke.xyzPos(t)
        coords[idx, :] = [x, y, z]
    return times, coords

def generate_walker_delta_constellation(
    total_sat=40,
    num_planes=5,
    altitude_km=570.0,
    inclination_deg=70.0,
    polar_inclination_deg=80.0,
    num_polar_planes=2,
    earth_radius_km=6371.0,
    e=0.0,
    arg_perigee_deg=0.0,
    F=1,
    Omega_0=0.0,
    M_0=0.0,
):
    """
    Walker-Δ 콘스텔레이션 생성
      - 처음 num_polar_planes은 polar_inclination_deg,
      - 나머지 평면은 inclination_deg를 사용
    """
    a = earth_radius_km + altitude_km
    sats_per_plane = total_sat // num_planes
    delta_RAAN = 360.0 / num_planes
    delta_M = 360.0 / sats_per_plane

    kepler_elements = []
    for p in range(num_planes):
        # 플레인 p의 경사각 결정
        inc = polar_inclination_deg if p < num_polar_planes else inclination_deg
        RAAN_p = Omega_0 + p * delta_RAAN
        for s in range(sats_per_plane):
            M_ps = (M_0 + s * delta_M +
                    p * (F * 360.0) / (num_planes * sats_per_plane))
            kepler_elements.append((a, e, inc, RAAN_p, arg_perigee_deg, M_ps))
    return kepler_elements

def generate_tle_from_kepler(a, e, i_deg, RAAN_deg, w_deg, M_deg, epoch_dt, satnum, mu=398600.4418):
    """
    Kepler 요소를 받아 TLE 두 줄 문자열을 생성합니다.
    아래는 TLE 형식의 각 필드(고정 폭)를 공식 규격에 맞게 포맷한 예제입니다.
    
    TLE Line 1 (총 69문자, 마지막 1문자는 체크섬):
      01: "1"
      02: space
      03-07: 위성번호 (5자리, 오른쪽 정렬)
      08: 분류 (예: "U")
      09: space
      10-17: 국제식별자 (8자리, 오른쪽 정렬)
      18: space
      19-32: Epoch (YYDDD.DDDDDDDD, 14자리, 오른쪽 정렬)
      33: space
      34-43: 첫 번째 평균 운동 미분값 (10자리, 오른쪽 정렬)
      44: space
      45-52: 두 번째 평균 운동 미분값 (8자리, 오른쪽 정렬)
      53: space
      54-61: BSTAR 드래그 계수 (8자리, 오른쪽 정렬)
      62: space
      63: Ephemeris type (1자리)
      64: space
      65-68: Element set 번호 (4자리, 오른쪽 정렬)
      69: 체크섬 (계산된 1자리 숫자)
    
    TLE Line 2:
      01: "2"
      02: space
      03-07: 위성번호 (5자리, 오른쪽 정렬)
      08: space
      09-16: 기울기 (deg, 8자리, 소수점 4자리)
      17: space
      18-25: RAAN (deg, 8자리, 소수점 4자리)
      26: space
      27-33: 이심률 (소수점 없이 7자리)
      34: space
      35-42: 근점인자 (deg, 8자리, 소수점 4자리)
      43: space
      44-51: 평균 이상 (deg, 8자리, 소수점 4자리)
      52: space
      53-63: 평균 운동 (rev/day, 11자리, 소수점 8자리)
      64-68: Revolution number at epoch (5자리)
      69: 체크섬
    """
    # 평균 운동 계산
    T = 2 * math.pi * math.sqrt(a**3 / mu)
    mean_motion = 86400.0 / T

    # Epoch: YYDDD.DDDDDDDD
    epoch_year = epoch_dt.year % 100
    day_of_year = (epoch_dt - datetime.datetime(epoch_dt.year, 1, 1)).total_seconds() / 86400.0 + 1
    # epoch_str는 총 14자리
    epoch_str = f"{epoch_year:02d}{day_of_year:012.8f}"
    
    # 각 필드 정의
    satnum_str    = f"{satnum:05d}"
    classification = "U"
    # 국제식별자는 예제에서는 "20000A"를 사용하고, 총 8자리 필드로 채웁니다.
    int_desig     = f"{'20000A':>8s}"
    first_deriv   = f"{'.00000000':>10s}"   # 10자리
    second_deriv  = f"{'00000-0':>8s}"       # 8자리 (부호 포함이 필요한 경우 수정)
    bstar         = f"{'00000-0':>8s}"       # 8자리
    ephemeris_type = "0"
    element_set   = f"{999:4d}"             # 4자리, 예제에서는 999 사용

    # TLE Line 1 구성 (총 68문자, 체크섬 제외)
    line1 = (
        "1 " +
        f"{satnum_str:>5s}" +
        classification +
        " " +
        f"{int_desig:>8s}" +
        " " +
        f"{epoch_str:>14s}" +
        " " +
        f"{first_deriv:>10s}" +
        " " +
        f"{second_deriv:>8s}" +
        " " +
        f"{bstar:>8s}" +
        " " +
        f"{ephemeris_type:1s}" +
        " " +
        f"{element_set:>4s}"
    )
    if len(line1) != 68:
        raise ValueError(f"Line1 길이({len(line1)})가 68가 아님:\n'{line1}'")
    checksum1 = compute_tle_checksum(line1)
    line1_full = line1 + str(checksum1)

    # 이심률: 소수점 없이 7자리 (예: 0.0000000 -> "0000000")
    ecc_str = f"{e:.7f}"[2:]

    # TLE Line 2 구성 (총 68문자, 체크섬 제외)
    line2 = (
        "2 " +
        f"{satnum_str:>5s}" +
        " " +
        f"{i_deg:8.4f}" +
        " " +
        f"{RAAN_deg:8.4f}" +
        " " +
        f"{ecc_str:7s}" +
        " " +
        f"{w_deg:8.4f}" +
        " " +
        f"{M_deg:8.4f}" +
        " " +
        f"{mean_motion:11.8f}" +
        "00000"  # Revolution number at epoch (5자리)
    )
    if len(line2) != 68:
        raise ValueError(f"Line2 길이({len(line2)})가 68가 아님:\n'{line2}'")
    checksum2 = compute_tle_checksum(line2)
    line2_full = line2 + str(checksum2)

    return line1_full, line2_full

def propagate_tle(tle_line1, tle_line2, start_dt, end_dt, step_minutes=1):
    sat = Satrec.twoline2rv(tle_line1, tle_line2, WGS72)
    total_minutes = int((end_dt - start_dt).total_seconds() / 60)
    steps = total_minutes // step_minutes + 1
    times = []
    coords = []
    for step in range(steps):
        current_dt = start_dt + datetime.timedelta(minutes=step * step_minutes)
        times.append(current_dt)
        jd, fr = jday(current_dt.year, current_dt.month, current_dt.day,
                      current_dt.hour, current_dt.minute, current_dt.second)
        e_code, r, v = sat.sgp4(jd, fr)
        if e_code == 0:
            coords.append(r)
        else:
            coords.append([np.nan, np.nan, np.nan])
    return np.array(times), np.array(coords)

def propagate_tle_constellation(kepler_elements, epoch_dt, t_end, step_minutes=1):
    constellation_results = []
    for idx, ke in enumerate(kepler_elements):
        a, e, i_deg, RAAN_deg, w_deg, M_deg = ke
        tle_line1, tle_line2 = generate_tle_from_kepler(a, e, i_deg, RAAN_deg, w_deg, M_deg, epoch_dt, idx+1)
        end_dt = epoch_dt + datetime.timedelta(seconds=t_end)
        times, coords = propagate_tle(tle_line1, tle_line2, epoch_dt, end_dt, step_minutes=step_minutes)
        sat_info = {"sat_id": idx, "tle": (tle_line1, tle_line2), "times": times, "coords": coords, "kepler": ke}
        constellation_results.append(sat_info)
    return constellation_results

def save_constellation_csv(constellation_result, filename="walker_delta_result.csv"):
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sat_id", "a(km)", "e", "i(deg)", "RAAN(deg)", "w(deg)", "M(deg)",
                         "time(s)", "x(km)", "y(km)", "z(km)"])
        for sat_info in constellation_result:
            sat_id = sat_info["sat_id"]
            a, e_, i_deg, RAAN_deg, w_deg, M_deg = sat_info["kepler"]
            times = sat_info["times"]
            coords = sat_info["coords"]
            for t, (x, y, z) in zip(times, coords):
                writer.writerow([sat_id, a, e_, i_deg, RAAN_deg, w_deg, M_deg,
                                 t, x, y, z])

def save_tle_constellation_csv(constellation_results, filename="tle_constellation.csv"):
    """
    위성군의 TLE 전파 결과를 CSV 파일로 저장합니다.
    """
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sat_id", "TLE_line1", "TLE_line2", "Datetime (UTC)", "X (km)", "Y (km)", "Z (km)"])
        for sat in constellation_results:
            sat_id = sat["sat_id"]
            tle_line1, tle_line2 = sat["tle"]
            for t, (x, y, z) in zip(sat["times"], sat["coords"]):
                writer.writerow([sat_id, tle_line1, tle_line2, t.isoformat(), x, y, z])

def plot_constellation_3d(
    total_sat,
    constellation_results,
    num_planes,
    num_polar_planes,
    sample_step=200,
    earth_radius=6371.0
):
    """
    - Earth: 반투명 구체 surface
    - 위성 궤도: scatter 점만 (marker='.', s=1)
    - Polar vs Non-polar 색 구분
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # ———— Earth surface ————
    u = np.linspace(0, 2 * np.pi, 60)
    v = np.linspace(0, np.pi, 30)
    x_e = earth_radius * np.outer(np.cos(u), np.sin(v))
    y_e = earth_radius * np.outer(np.sin(u), np.sin(v))
    z_e = earth_radius * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(
        x_e, y_e, z_e,
        rstride=2, cstride=2,
        color='lightblue',
        alpha=0.3,
        linewidth=0
    )

    # ———— satellite scatter ————
    sats_per_plane = total_sat // num_planes
    for idx, sat in enumerate(constellation_results):
        plane_idx = idx // sats_per_plane
        is_polar = (plane_idx < num_polar_planes)

        # sample to reduce points
        pts = sat["coords"][::sample_step]
        xs, ys, zs = pts[:,0], pts[:,1], pts[:,2]

        if is_polar:
            ax.scatter(xs, ys, zs,
                       c='C3', s=2, marker='.', alpha=0.6,
                       label="Polar planes" if plane_idx==0 else None)
        else:
            ax.scatter(xs, ys, zs,
                       c='C1', s=2, marker='.', alpha=0.8,
                       label="Non-polar planes" if plane_idx==num_polar_planes else None)

    # ———— styling ————
    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")
    ax.set_zlabel("Z (km)")
    ax.set_title("Walker Delta Constellation 3D")

    # legend dedupe
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=8)

    # ensure equal aspect
    max_range = earth_radius + 2000
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.set_box_aspect([1,1,1])
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_zlim(-max_range, max_range)

    plt.tight_layout()
    plt.show()
# ----------------------------------------------------
# 통신 창 세부 정보 계산 (elevation 창 구간 계산)
# ----------------------------------------------------
def get_contiguous_windows(times, valid_flags, step_minutes=1):
    """
    Boolean 배열 valid_flags에 대해 연속 구간(창)을 찾아,
    각 구간의 시작, 종료 시간과 지속 시간을 (분) 반환합니다.
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

def compute_communication_window_details(constellation_results, observer, threshold_deg=40, step_minutes=1):
    """
    각 위성에 대해 주어진 observer (Topos 객체)에서의 elevation을 계산하여,
    elevation이 (90 - threshold_deg) 이상인 통신 창의 시작/종료 시간을 상세하게 계산합니다.
    """
    ts = load.timescale()
    details = []
    for sat in constellation_results:
        tle_line1, tle_line2 = sat["tle"]
        sat_name = f"Sat{sat['sat_id']}"
        satellite = EarthSatellite(tle_line1, tle_line2, sat_name, ts)
        times_dt = sat["times"]
        t_sf = ts.utc([dt.year for dt in times_dt],
                      [dt.month for dt in times_dt],
                      [dt.day for dt in times_dt],
                      [dt.hour for dt in times_dt],
                      [dt.minute for dt in times_dt],
                      [dt.second for dt in times_dt])
        difference = satellite - observer
        topocentric = difference.at(t_sf)
        alt, az, distance = topocentric.altaz()
        elev = alt.degrees
        valid = elev >= threshold_deg
        windows = get_contiguous_windows(times_dt, valid, step_minutes=step_minutes)
        total_minutes = sum([w[2] for w in windows])
        details.append({"sat_id": sat["sat_id"], "windows": windows, "total_minutes": total_minutes})
    return details

def compute_communication_windows(constellation_results, ground_station, threshold_deg=10):
    """
    주어진 지상국(Topos 객체)에서, 각 위성에 대해 주어진 시간 동안 elevation이 threshold_deg 이상인 
    시점(통신 가능 시간)을 계산합니다.
    """
    ts = load.timescale()
    comm_dict = {}
    for sat in constellation_results:
        tle_line1, tle_line2 = sat["tle"]
        sat_name = f"Sat{sat['sat_id']}"
        satellite = EarthSatellite(tle_line1, tle_line2, sat_name, ts)
        times_dt = sat["times"]
        t_sf = ts.utc([dt.year for dt in times_dt],
                      [dt.month for dt in times_dt],
                      [dt.day for dt in times_dt],
                      [dt.hour for dt in times_dt],
                      [dt.minute for dt in times_dt],
                      [dt.second for dt in times_dt])
        observer = ground_station
        difference = satellite - observer
        topocentric = difference.at(t_sf)
        alt, az, distance = topocentric.altaz()
        elev = alt.degrees
        valid = elev >= threshold_deg
        total_minutes = np.sum(valid) * 1  # 1분 간격
        comm_dict[sat_name] = total_minutes
    return comm_dict

def save_comm_windows_csv(comm_details, ground_station_name, filename="communication_windows.csv"):
    """
    통신 창 상세 정보를 CSV 파일로 저장합니다.
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

# ----------------------------------------------------
# Berlin 지상국 기준 위성 고도각(elevation) vs 시간 플롯 함수
# ----------------------------------------------------
def plot_elevation_vs_time(constellation_results, observer, title="Elevation vs Time (Berlin)"):
    """
    주어진 위성 전파 결과(constellation_results)를 기반으로,
    observer (Topos 객체) 위치에서 위성의 elevation(고도각)을 시간에 따라 플롯합니다.
    """
    ts = load.timescale()
    plt.figure(figsize=(12, 6))
    for sat in constellation_results:
        tle_line1, tle_line2 = sat["tle"]
        sat_name = f"Sat{sat['sat_id']}"
        satellite = EarthSatellite(tle_line1, tle_line2, sat_name, ts)
        times_dt = sat["times"]
        t_sf = ts.utc([dt.year for dt in times_dt],
                      [dt.month for dt in times_dt],
                      [dt.day for dt in times_dt],
                      [dt.hour for dt in times_dt],
                      [dt.minute for dt in times_dt],
                      [dt.second for dt in times_dt])
        difference = satellite - observer
        topocentric = difference.at(t_sf)
        alt, az, distance = topocentric.altaz()
        elev = alt.degrees
        plt.plot(times_dt, elev, alpha=0.7, label=sat_name)
    plt.xlabel("Time (UTC)")
    plt.ylabel("Elevation (°)")
    plt.title(title)
    plt.legend(fontsize=8, loc="upper right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def save_tle_file(constellation_results, filename="constellation.tle"):
    """
    각 위성에 대해 이름 줄(예: SAT00001)과 TLE 2줄을 저장합니다.
    """
    with open(filename, "w") as f:
        for sat in constellation_results:
            sat_id = sat["sat_id"]
            tle_line1, tle_line2 = sat["tle"]
            f.write(f"SAT{sat_id:05d}\n")
            f.write(tle_line1.rstrip() + "\n")
            f.write(tle_line2.rstrip() + "\n")
    print(f"TLE 정보가 '{filename}' 파일로 저장되었습니다.")
    
def save_elevation_csv(constellation_results, observer, filename="elevations.csv"):
    """
    각 위성의 시간별 elevation (고도각)을 CSV 파일로 저장합니다.
    
    CSV 컬럼: sat_id, Datetime (UTC), Elevation (deg)
    
    Parameters:
      constellation_results: 각 위성의 전파 결과 리스트 (각 항목에 "tle", "times" 등 포함)
      observer: Skyfield Topos 객체 (예: Berlin 지상국)
      filename: 저장할 CSV 파일 이름 (기본: "elevations.csv")
    """
    ts = load.timescale()
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sat_id", "Datetime (UTC)", "Elevation (deg)"])
        
        for sat in constellation_results:
            tle_line1, tle_line2 = sat["tle"]
            sat_name = f"Sat{sat['sat_id']}"
            satellite = EarthSatellite(tle_line1, tle_line2, sat_name, ts)
            times_dt = sat["times"]
            t_sf = ts.utc(
                [dt.year for dt in times_dt],
                [dt.month for dt in times_dt],
                [dt.day for dt in times_dt],
                [dt.hour for dt in times_dt],
                [dt.minute for dt in times_dt],
                [dt.second for dt in times_dt]
            )
            difference = satellite - observer
            topocentric = difference.at(t_sf)
            alt, az, distance = topocentric.altaz()
            elev = alt.degrees
            for time_dt, elev_deg in zip(times_dt, elev):
                writer.writerow([sat["sat_id"], time_dt.isoformat(), elev_deg])
    
    print(f"Elevation 데이터가 '{filename}' 파일로 저장되었습니다.")

# ----------------------------------------------------
# Main 실행
# ----------------------------------------------------
if __name__ == "__main__":
    # 변수값
    total_sat           = 200
    num_planes          = 20
    num_polar_planes    = 2
    base_inclination    = 70.0
    polar_inclination   = 80.0
    altitude_km         = 570.0
    F                   = 1
    t_end               = 1209600.0  # 2주
    step_minutes        = 1

    ground_stations = [
        {'name': 'Houston', 'lat': 29.76, 'lon': -95.37, 'elevation_m': 30},
        {'name': 'Berlin', 'lat': 52.52, 'lon': 13.41, 'elevation_m': 34},
        {'name': 'Tokyo', 'lat': 35.68, 'lon': 139.69, 'elevation_m': 40},
        {'name': 'Nairobi', 'lat': -1.29, 'lon': 36.82, 'elevation_m': 1700},
        {'name': 'Sydney', 'lat': -33.87, 'lon': 151.21, 'elevation_m': 58},
    ]
    iot_regions = [
        {'name': 'Abisko', 'lat': 68.35, 'lon': 18.79, 'elevation_m': 420},
        {'name': 'Boreal', 'lat': 55.50, 'lon': 105.00, 'elevation_m': 450},
        {'name': 'Taiga', 'lat': 58.00, 'lon': 99.00, 'elevation_m': 300},
        {'name': 'Patagonia', 'lat': 51.00, 'lon': 73.00, 'elevation_m': 500},
        {'name': 'Amazon_Forest', 'lat': -3.47, 'lon': -62.37, 'elevation_m': 100},  # 아마존 열대우림
        {'name': 'Great_Barrier', 'lat': -18.29, 'lon': 147.77, 'elevation_m': 0},   # 그레이트 배리어 리프
        {'name': 'Mediterranean', 'lat': 37.98, 'lon': 23.73, 'elevation_m': 170},    # 지중해 연안
        {'name': 'California', 'lat': 36.78, 'lon': -119.42, 'elevation_m': 150},
    ]
    # Berlin 지상국 (두 번째 항목)을 observer로 사용
    gs = ground_stations[1]
    # IoT 지역 설정
    iot = iot_regions[3]
    # boreal = iot_regions[1]
    # taiga = iot_regions[2]
    # patagonia = iot_regions[3]

    # Walker Delta 방식으로 위성군의 Kepler 요소 생성
    kepler_elements = generate_walker_delta_constellation(
        total_sat=total_sat,
        num_planes=num_planes,
        altitude_km=altitude_km,
        inclination_deg=base_inclination,
        polar_inclination_deg=polar_inclination,
        num_polar_planes=num_polar_planes,
        F=F
    )

    # TLE epoch: 예시로 2021년 3월 2일 00:00:00 UTC
    epoch_dt = datetime.datetime(2025, 3, 30, 0, 0, 0)

    # TLE 기반 위성군 전파 (step_minutes=1 → t_end 기간 동안)

    constellation_results = propagate_tle_constellation(
        kepler_elements,
        epoch_dt,
        t_end,
        step_minutes=step_minutes
    )
    print(f"총 {len(constellation_results)}개 위성 전파 결과 생성됨 (TLE 기반)")

    # TLE 파일로 저장
    save_tle_file(constellation_results)

    # 3D 플롯으로 위성군 전파 결과 시각화
    plot_constellation_3d(total_sat = total_sat, constellation_results = constellation_results,
                          num_planes = num_planes, num_polar_planes = num_polar_planes)

    # CSV 파일로 위성 전파 결과 저장
    save_tle_constellation_csv(constellation_results)
    print("위성 전파 결과 CSV 파일 저장 완료.")

    # 필요에 따라 아래 코드의 주석을 해제하여 추가 플롯이나 통신 창 계산을 진행할 수 있습니다.
    # 예) Berlin 지상국 기준 위성 고도각 플롯:
    threshold_deg = 40
    gs_topos = Topos(latitude_degrees=gs.get('lat'),
                         longitude_degrees=gs.get('lon'),
                         elevation_m=gs.get('elevation_m'))
    # IoT 센서 클러스터
    # iot_topos = Topos(latitude_degrees=iot.get('lat'),
    #                      longitude_degrees=iot.get('lon'),
    #                     elevation_m=iot.get('elevation_m'))
    
    save_elevation_csv(constellation_results, gs_topos, filename="elevations_berlin.csv")
    # plot_elevation_vs_time(constellation_results, gs_topos, title="Elevation vs Time from Berlin")

    comm_details_berlin = compute_communication_window_details(constellation_results, gs_topos, threshold_deg=threshold_deg, step_minutes=1)
    # comm_details_iot = compute_communication_window_details(constellation_results, iot_topos, threshold_deg=threshold_deg, step_minutes=1)
    total_comm_berlin = sum([d["total_minutes"] for d in comm_details_berlin])
    save_comm_windows_csv(comm_details_berlin, ground_station_name=gs['name'], filename="comm_windows_" + gs['name'] + ".csv")
    # save_comm_windows_csv(comm_details_iot, ground_station_name=iot['name'], filename="comm_windows_" + iot['name'] + ".csv")
    print(f"\n{gs['name']} 지상국에서 전체 위성과의 통신 가능 총 시간: {total_comm_berlin:.1f} 분")

    # 총 통신 가능 시간 출력 (IoT 센서 클러스터)
    # total_comm_iot = sum([d["total_minutes"] for d in comm_details_iot])
    # print(f"{iot['name']} IoT 센서 클러스터에서 전체 위성과의 통신 가능 총 시간: {total_comm_iot:.1f} 분")

    print("\n각 IoT 클러스터와 위성군 간 통신 시간:")
    for iot in iot_regions:
        iot_topos = Topos(
            latitude_degrees=iot['lat'],
            longitude_degrees=iot['lon'],
            elevation_m=iot['elevation_m']
        )
        
        # 각 IoT 클러스터에 대한 통신 창 계산
        comm_details = compute_communication_window_details(
            constellation_results, 
            iot_topos, 
            threshold_deg=threshold_deg
        )
        
        # 총 통신 가능 시간 계산
        total_comm_time = sum([d["total_minutes"] for d in comm_details])
        print(f"{iot['name']}: {total_comm_time:.1f} 분")
        
        # 통신 창 상세 정보 저장
        save_comm_windows_csv(
            comm_details, 
            ground_station_name=iot['name'], 
            filename=f"comm_windows_{iot['name']}.csv"
        )
