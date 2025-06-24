import numpy as np
import datetime
from sgp4.api import Satrec, WGS72, jday
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import csv
from PyAstronomy import pyasl
from skyfield.api import load, EarthSatellite, Topos


# ===============================
# 기존 TLE 기반 위성군 전파 관련 함수들
# ===============================
def propagate_leo(a, per, e, tau, Omega, w, i, num_points=200):
    ke = pyasl.KeplerEllipse(a, per, e=e, tau=tau, Omega=Omega, w=w, i=i)
    times = np.linspace(0, per, num_points)
    coords = np.zeros((num_points, 3), dtype=float)
    for idx, t in enumerate(times):
        x, y, z = ke.xyzPos(t)
        coords[idx, :] = [x, y, z]
    return times, coords

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

def generate_tle_from_kepler(a, e, i_deg, RAAN_deg, w_deg, M_deg, epoch_dt, satnum, mu=398600.4418):
    # import math
    # T = 2 * math.pi * math.sqrt(a**3 / mu)
    # mean_motion = 86400.0 / T  # rev/day
    # # Epoch 생성: "YYDDD.DDDDDDDD" 형식 (총 14자리)
    # epoch_year = epoch_dt.year % 100
    # start_of_year = datetime.datetime(epoch_dt.year, 1, 1)
    # day_of_year = (epoch_dt - start_of_year).total_seconds() / 86400.0 + 1
    # day_int = int(day_of_year)
    # frac = day_of_year - day_int
    # frac_int = int(round(frac * 1e8))
    # epoch_str = f"{epoch_year:02d}{day_int:03d}.{frac_int:08d}"  # 예: "21061.00000000"
    # line1 = f"1 {satnum:05d}U 20000A {epoch_str}  0.00000000  00000-0  00000-0 0  9999"
    # if len(line1) < 69:
    #     line1 = line1.ljust(69)
    # ecc_str = f"{e:.7f}"[2:]
    # line2 = (f"2 {satnum:05d} "
    #          f"{i_deg:8.4f} {RAAN_deg:8.4f} {ecc_str:7s} "
    #          f"{w_deg:8.4f} {M_deg:8.4f} {mean_motion:11.8f}00000")
    # line2 = line2.ljust(68) + "0"
    import math
    T = 2 * math.pi * math.sqrt(a**3 / mu)
    mean_motion = 86400.0 / T

    epoch_year = epoch_dt.year % 100
    day_of_year = (epoch_dt - datetime.datetime(epoch_dt.year, 1, 1)).total_seconds() / 86400.0 + 1
    
    # 정확한 TLE epoch 포맷: 소수점 뒤 8자리 (DDD.DDDDDDDD)
    epoch_str = f"{epoch_year:02d}{day_of_year:012.8f}"

    # TLE Line 1
    line1 = f"1 {satnum:05d}U 20000A   {epoch_str}  .00000000  00000-0  00000-0 0  9990"
    line1 = line1.ljust(69)

    ecc_str = f"{e:.7f}"[2:]
    
    # TLE Line 2
    line2 = (f"2 {satnum:05d} {i_deg:8.4f} {RAAN_deg:8.4f} {ecc_str:7s} "
             f"{w_deg:8.4f} {M_deg:8.4f} {mean_motion:11.8f}00000")
    line2 = line2.ljust(68) + "0"

    return line1, line2

def propagate_tle(tle_line1, tle_line2, start_dt, end_dt, step_minutes=1):
    sat = Satrec.twoline2rv(tle_line1, tle_line2, whichconst=WGS72)
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
        sat_info = {"sat_id": idx, "tle": (tle_line1, tle_line2), "times": times, "coords": coords}
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

def plot_constellation_3d(constellation_result, show_legend=True):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    for sat in constellation_result:
        times = sat["times"]
        coords = sat["coords"]
        x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
        if show_legend:
            ax.plot(x, y, z, label=f"sat {sat['sat_id']}")
        else:
            ax.plot(x, y, z)
    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")
    ax.set_zlabel("Z (km)")
    ax.set_title("Walker Delta Constellation 3D Propagation")
    if show_legend:
        ax.legend(fontsize=8, loc="upper right")
    plt.show()

# ===============================================
# 통신 창 세부 정보 계산 (elevation 창 구간 계산)
# ===============================================
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

def compute_communication_window_details(constellation_results, observer, threshold_deg=10, step_minutes=1):
    """
    각 위성에 대해 주어진 observer (Topos 객체)에서의 elevation을 계산하여,
    elevation이 threshold_deg 이상인 통신 창의 시작/종료 시간을 상세하게 계산합니다.
    
    반환:
      details: 리스트 of dict, 각 dict는
         {"sat_id": int, "windows": [(start_time, end_time, duration_min), ...], "total_minutes": float }
    """
    ts = load.timescale()
    details = []
    for sat in constellation_results:
        tle_line1, tle_line2 = sat["tle"]
        sat_name = f"Sat{sat['sat_id']}"
        satellite = EarthSatellite(tle_line1, tle_line2, sat_name, ts)
        # 변환: sat["times"] (datetime array) -> Skyfield times
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
        valid = elev >= (90 - threshold_deg)
        windows = get_contiguous_windows(times_dt, valid, step_minutes=step_minutes)
        total_minutes = sum([w[2] for w in windows])
        details.append({"sat_id": sat["sat_id"], "windows": windows, "total_minutes": total_minutes})
    return details

def compute_communication_windows(constellation_results, ground_station, threshold_deg=10):
    """
    주어진 지상국(Topos 객체)에서, 각 위성에 대해 주어진 시간 동안 elevation이 threshold_deg 이상인 
    시점(통신 가능 시간)을 계산합니다.
    
    반환:
      결과 딕셔너리: { sat_id: total_communication_minutes }
    """
    ts = load.timescale()
    comm_dict = {}
    # 시간 배열을 Skyfield timescale으로 변환 (여기서는 constellation_results 내 times가 datetime 객체임)
    for sat in constellation_results:
        tle_line1, tle_line2 = sat["tle"]
        sat_name = f"Sat{sat['sat_id']}"
        satellite = EarthSatellite(tle_line1, tle_line2, sat_name, ts)
        # 변환: constellation_results의 times는 datetime 배열
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
        # 통신 가능한 시점 수
        valid = elev >= threshold_deg
        # 각 시점 간격은 step_minutes (분) : 여기서는 1분 간격
        total_minutes = np.sum(valid) * 1  # 1분씩 계산
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

if __name__ == "__main__":
    # 변수값
    total_sat = 60
    num_planes = 5
    altitude_km = 570.0
    inclination_deg = 70.0
    F = 1
    t_end = 1209600.0  # 초
    num_points = 100  # 시뮬레이션 시점

    ground_stations = [
        {'name': 'Houston', 'lat': 29.76, 'lon': -95.37, 'elevation_m': 30},
        {'name': 'Berlin', 'lat': 52.52, 'lon': 13.41, 'elevation_m': 34},
        {'name': 'Tokyo', 'lat': 35.68, 'lon': 139.69, 'elevation_m': 40},
        {'name': 'Nairobi', 'lat': -1.29, 'lon': 36.82, 'elevation_m': 1700},
        {'name': 'Sydney', 'lat': -33.87, 'lon': 151.21, 'elevation_m': 58},
    ]
    berlin = ground_stations[1]

    # Walker Delta 방식으로 위성군의 Kepler 요소 생성
    kepler_elements = generate_walker_delta_constellation(
        total_sat=total_sat,
        num_planes=num_planes,
        altitude_km=altitude_km,
        inclination_deg=inclination_deg,
        F=F
    )

    # TLE epoch: 예시로 2021년 3월 2일 00:00:00 UTC
    epoch_dt = datetime.datetime(2021, 3, 2, 0, 0, 0)

    # TLE 기반 위성군 전파 (step_minutes=1 → 약 100 시점)
    constellation_results = propagate_tle_constellation(kepler_elements, epoch_dt, t_end, step_minutes=1)
    # print(constellation_results)
    print(f"총 {len(constellation_results)}개 위성 전파 결과 생성됨 (TLE 기반)")

    # 3D 플롯으로 결과 시각화
    plot_constellation_3d(constellation_results, show_legend=False)

    # CSV 파일로 위성 전파 결과 저장
    save_tle_constellation_csv(constellation_results)
    print("위성 전파 결과 CSV 파일 저장 완료.")

    # ===============================================
    # 지상국 및 IoT 센서 위치 정의 (Topos 객체 사용)
    # ===============================================
    # 지상국 (예: Houston)
    from skyfield.api import Topos
    gs = Topos(latitude_degrees=berlin.get('lat'), longitude_degrees=berlin.get('lon'), elevation_m=berlin.get('elevation_m'))
    # IoT 센서 클러스터 (예: Amazon_Manaus)
    iot = Topos(latitude_degrees=-3.1, longitude_degrees=-60.0, elevation_m=50)
    threshold_deg = 10

    # 통신 창 상세 정보 계산 (각 지상국별로 계산 가능)
    comm_details_gs = compute_communication_window_details(constellation_results, gs, threshold_deg=threshold_deg, step_minutes=1)
    comm_details_iot = compute_communication_window_details(constellation_results, iot, threshold_deg=threshold_deg, step_minutes=1)

    # 총 통신 가능 시간 출력 (지상국)
    total_comm_gs = sum([d["total_minutes"] for d in comm_details_gs])
    print(f"\nBerlin 지상국에서 전체 위성과의 통신 가능 총 시간: {total_comm_gs:.1f} 분")
    # 총 통신 가능 시간 출력 (IoT 센서 클러스터)
    total_comm_iot = sum([d["total_minutes"] for d in comm_details_iot])
    print(f"Amazon_Manaus IoT 센서 클러스터에서 전체 위성과의 통신 가능 총 시간: {total_comm_iot:.1f} 분")

    # CSV 파일로 통신 창 상세 정보 저장
    save_comm_windows_csv(comm_details_gs, ground_station_name=berlin.get('name'), filename="comm_windows_Berlin.csv")
    save_comm_windows_csv(comm_details_iot, ground_station_name="Amazon_Manaus", filename="comm_windows_Amazon_Manaus.csv")
    print("통신 창 상세 정보 CSV 파일 저장 완료.")