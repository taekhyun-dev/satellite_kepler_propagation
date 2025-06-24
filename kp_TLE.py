import numpy as np
import datetime
from sgp4.api import Satrec, WGS72, jday
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import csv

def generate_walker_delta_constellation(
    total_sat=100,
    num_planes=20,
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
    Walker Delta 구성을 모방한 LEO 위성군의 Kepler 요소를 생성합니다.
    
    반환값:
      위성별 Kepler 요소 튜플 (a, e, i, RAAN, w, M)의 리스트.
    """
    a = earth_radius_km + altitude_km
    sats_per_plane = total_sat // num_planes
    delta_RAAN = 360.0 / num_planes
    delta_M = 360.0 / sats_per_plane

    kepler_elements = []
    for p in range(num_planes):
        for s in range(sats_per_plane):
            RAAN_p = Omega_0 + p * delta_RAAN
            M_ps = (M_0
                    + s * delta_M
                    + p * (F * 360.0) / (num_planes * sats_per_plane))
            ke = (a, e, inclination_deg, RAAN_p, arg_perigee_deg, M_ps)
            kepler_elements.append(ke)

    return kepler_elements

def generate_tle_from_kepler(a, e, i_deg, RAAN_deg, w_deg, M_deg, epoch_dt, satnum, mu=398600.4418):
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
    """
    SGP4 라이브러리를 사용하여 단일 TLE의 전파 결과(시간별 (x,y,z))를 계산합니다.
    """
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
    """
    각 위성에 대해 Kepler 요소로부터 TLE를 생성한 후, 0 ~ t_end (초) 구간의 전파 결과를 반환합니다.
    
    반환값: 각 위성의 결과를 담은 리스트 (dict 형태)
      { "sat_id": int, "tle": (line1, line2), "times": datetime array, "coords": (N,3) array }
    """
    constellation_results = []
    for idx, ke in enumerate(kepler_elements):
        a, e, i_deg, RAAN_deg, w_deg, M_deg = ke
        tle_line1, tle_line2 = generate_tle_from_kepler(a, e, i_deg, RAAN_deg, w_deg, M_deg, epoch_dt, idx+1)
        end_dt = epoch_dt + datetime.timedelta(seconds=t_end)
        times, coords = propagate_tle(tle_line1, tle_line2, epoch_dt, end_dt, step_minutes=step_minutes)
        sat_info = {
            "sat_id": idx,
            "tle": (tle_line1, tle_line2),
            "times": times,
            "coords": coords
        }
        constellation_results.append(sat_info)
    return constellation_results

def plot_tle_constellation_3d(constellation_results, show_legend=False):
    """
    위성군의 TLE 전파 결과를 3D 플롯으로 시각화합니다.
    """
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')
    for sat in constellation_results:
        coords = sat["coords"]
        x, y, z = coords[:,0], coords[:,1], coords[:,2]
        ax.plot(x, y, z, label=f"sat {sat['sat_id']}")
    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")
    ax.set_zlabel("Z (km)")
    ax.set_title("TLE-based Constellation 3D Propagation")
    if show_legend:
        ax.legend(fontsize=8)
    plt.show()

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

# -------------------------------
# 메인 실행부: 요청한 변수값으로 구성
# -------------------------------
if __name__ == "__main__":
    # 요청 변수값
    total_sat = 100
    num_planes = 20
    altitude_km = 570.0
    inclination_deg = 70.0
    F = 1
    t_end = 6000.0  # 초
    
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
    print(f"총 {len(constellation_results)}개 위성 전파 결과 생성됨 (TLE 기반)")
    
    # 3D 플롯으로 결과 시각화
    plot_tle_constellation_3d(constellation_results, show_legend=False)
    
    # CSV 파일로 저장
    save_tle_constellation_csv(constellation_results, filename="tle_constellation.csv")
    print("CSV 파일 저장 완료.")
