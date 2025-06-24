import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PyAstronomy import pyasl
import csv

def propagate_leo(a, per, e, tau, Omega, w, i, num_points=200):
    """
    KeplerEllipse를 사용하여 주어진 궤도 요소에 따른 위성 궤적을 계산합니다.
    
    매개변수:
      a       : 반장축 (km)
      per     : 궤도 주기 (초)
      e       : 이심률 (0~1, 여기서는 near circular)
      tau     : 근일점 통과 시각 (per와 동일 단위; t=0에서 근일점이면 0)
      Omega   : 승교점 적경 (deg)
      w       : 근일점 인자 (deg)
      i       : 경사각 (deg)
      num_points : 궤도 전파에 사용할 시점의 개수
      
    반환값:
      times  : 전파 시각 배열 (초)
      coords : 각 시각에서의 [x, y, z] 좌표 (km 단위, shape=(num_points, 3))
    """
    # KeplerEllipse 객체 생성 (Tref는 지원되지 않으므로 tau로 근일점 통과 시점을 지정)
    ke = pyasl.KeplerEllipse(a, per, e=e, tau=tau, Omega=Omega, w=w, i=i)
    
    # 0부터 per까지 num_points개의 시점 생성
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
    """
    Walker Delta 구성을 모방한 LEO 위성군의 케플러 요소를 생성합니다.
    
    매개변수:
      total_sat       : 전체 위성 수 T
      num_planes      : 궤도면 수 P
      altitude_km     : 위성 고도 (km)
      inclination_deg : 궤도 경사각 (deg)
      earth_radius_km : 지구 반지름 (km)
      e               : 이심률 (거의 0)
      arg_perigee_deg : 근일점 인자 (deg) - 원형 궤도에서는 0
      F               : Walker Delta phasing factor
      Omega_0         : RAAN의 기준값 (deg)
      M_0             : 평균 anomaly의 기준값 (deg)
      
    반환값:
      A list of tuples (a, e, i, RAAN, w, M)
      - a     : 반장축 (km)
      - e     : 이심률
      - i     : 경사각 (deg)
      - RAAN  : 승교점 적경 (deg)
      - w     : 근일점 인자 (deg)
      - M     : 평균 anomaly (deg)
    """
    
    # 반장축 (km) = 지구 반지름 + 위성 고도
    a = earth_radius_km + altitude_km
    
    # 궤도면당 위성 수
    sats_per_plane = total_sat // num_planes
    
    # RAAN 간격 (deg)
    delta_RAAN = 360.0 / num_planes
    
    # 평균 anomaly 간격 (deg)
    delta_M = 360.0 / sats_per_plane
    
    kepler_elements = []
    
    # p = 0..(num_planes-1), s = 0..(sats_per_plane-1)
    for p in range(num_planes):
        for s in range(sats_per_plane):
            # 승교점 적경
            RAAN_p = Omega_0 + p * delta_RAAN
            
            # Walker Delta 평균 anomaly
            # M_{p,s} = M_0 + (s * 360/S) + (p * F * 360/(P*S))
            # S = sats_per_plane
            M_ps = (M_0
                    + s * delta_M
                    + p * (F * 360.0) / (num_planes * sats_per_plane))
            
            # (a, e, i, RAAN, w, M)
            ke = (a, e, inclination_deg, RAAN_p, arg_perigee_deg, M_ps)
            kepler_elements.append(ke)
    
    return kepler_elements

from PyAstronomy import pyasl

def propagate_satellite(a, e, i_deg, RAAN_deg, w_deg, M_deg,
                        t_end, num_points=50, mu=398600.4418):
    """
    단일 위성의 케플러 요소로부터, PyAstronomy의 KeplerEllipse를 사용해
    0 ~ t_end 구간의 (x,y,z)를 전파합니다.
    
    매개변수:
      a         : 반장축 (km)
      e         : 이심률
      i_deg     : 경사각 (deg)
      RAAN_deg  : 승교점 적경 (deg)
      w_deg     : 근일점 인자 (deg)
      M_deg     : 평균 anomaly (deg)
      t_end     : 전파 종료 시각 (초)
      num_points: 시뮬레이션 시점 개수
      mu        : 지구 중력상수 (km^3/s^2)
      
    반환값:
      times  : 0 ~ t_end 사이의 시간 배열 (초)
      coords : (num_points, 3) 크기의 (x,y,z) 배열 (km)
    """
    # 1) 궤도 주기 (초) 계산: T = 2*pi * sqrt(a^3 / mu)
    T = 2.0 * np.pi * np.sqrt(a**3 / mu)
    
    # 2) 평균 anomaly (deg)를 rad로 변환
    M_rad = np.radians(M_deg)  # deg -> rad
    
    # 3) tau 계산 (PyAstronomy는 tau를 "근일점 통과 시점"으로 사용)
    #    M(t) = n(t - tau),  M(0)=M_rad => tau = - M_rad/n = -(M_rad/(2*pi))*T
    tau = -(M_rad / (2.0 * np.pi)) * T
    
    # 4) KeplerEllipse 객체 생성 (i, Omega, w는 deg 단위)
    ke = pyasl.KeplerEllipse(a, T, e=e, tau=tau,
                            Omega=RAAN_deg, w=w_deg, i=i_deg)
    
    # 5) 0 ~ t_end 구간을 num_points로 분할
    times = np.linspace(0, t_end, num_points)
    coords = np.zeros((num_points, 3), dtype=float)
    
    for idx, t in enumerate(times):
        # xyzPos(t): t초에서의 (x,y,z) [km]
        x, y, z = ke.xyzPos(t)
        coords[idx, :] = [x, y, z]
    
    return times, coords

def propagate_walker_delta_constellation(
    total_sat=20,
    num_planes=5,
    altitude_km=570.0,
    inclination_deg=70.0,
    earth_radius_km=6371.0,
    e=0.0,
    arg_perigee_deg=0.0,
    F=1,
    Omega_0=0.0,
    M_0=0.0,
    t_end=6000.0,
    num_points=10,
    mu=398600.4418
):
    """
    Walker Delta 구성으로 만들어진 LEO 위성군(720개 등)에 대하여,
    0 ~ t_end (초) 구간의 궤도 전파 결과 (x,y,z)를 계산.
    
    매개변수:
      - (Walker Delta 파라미터들) ...
      - t_end     : 전파 종료 시각 (초)
      - num_points: 시뮬레이션 시점 개수
      - mu        : 지구 중력상수
      
    반환값:
      constellation_result: 길이 total_sat인 리스트.
        각 원소는 dict 형태로, 
        {
          "sat_id": int,
          "kepler": (a, e, i, RAAN, w, M),
          "times": (num_points,) array,
          "coords": (num_points, 3) array
        }
    """
    # 1) 위성군 케플러 요소 생성
    ke_list = generate_walker_delta_constellation(
        total_sat, num_planes, altitude_km, inclination_deg,
        earth_radius_km, e, arg_perigee_deg, F, Omega_0, M_0
    )
    
    constellation_result = []
    
    # 2) 각 위성에 대해 propagate_satellite 호출
    for sat_id, ke in enumerate(ke_list):
        a, e_, i_deg, RAAN_deg, w_deg, M_deg = ke
        
        times, coords = propagate_satellite(
            a, e_, i_deg, RAAN_deg, w_deg, M_deg,
            t_end=t_end, num_points=num_points, mu=mu
        )
        
        sat_info = {
            "sat_id": sat_id,
            "kepler": ke,
            "times": times,
            "coords": coords
        }
        constellation_result.append(sat_info)
    
    return constellation_result

def save_constellation_csv(constellation_result, filename="walker_delta_result.csv"):
    """
    Walker Delta 전파 결과를 CSV 파일로 저장합니다.
    
    매개변수:
      constellation_result : propagate_walker_delta_constellation 함수 결과 (list of dict)
         각 dict는 {
           "sat_id": int,
           "kepler": (a, e, i_deg, RAAN_deg, w_deg, M_deg),
           "times": (num_points,) array,
           "coords": (num_points, 3) array
         }
      filename : 저장할 CSV 파일 이름
    """
    # CSV 모드로 파일 열기
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        
        # 헤더 작성
        writer.writerow(["sat_id", 
                         "a(km)", "e", "i(deg)", "RAAN(deg)", "w(deg)", "M(deg)",
                         "time(s)", "x(km)", "y(km)", "z(km)"])
        
        # 위성별 전파 결과를 순회하며 행(row) 작성
        for sat_info in constellation_result:
            sat_id = sat_info["sat_id"]
            a, e_, i_deg, RAAN_deg, w_deg, M_deg = sat_info["kepler"]
            times = sat_info["times"]
            coords = sat_info["coords"]
            
            # 시간 배열과 좌표 배열을 함께 순회
            for t, (x, y, z) in zip(times, coords):
                writer.writerow([sat_id, a, e_, i_deg, RAAN_deg, w_deg, M_deg,
                                 t, x, y, z])

def plot_constellation_3d(constellation_result, show_legend=True):
    """
    Walker Delta 위성군의 전파 결과를 3D 플롯으로 시각화합니다.
    
    매개변수:
      constellation_result : list of dict
          propagate_walker_delta_constellation() 함수 결과와 같은 구조의 리스트.
          각 dict는 {"sat_id": int, "kepler": (a, e, i, RAAN, w, M), "times": array, "coords": array} 형태입니다.
      show_legend : bool, optional
          위성 레이블(예: sat_id)을 표시할지 여부 (위성 수가 많으면 False 권장).
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    for sat in constellation_result:
        coords = sat["coords"]  # shape = (num_points, 3)
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

# if __name__ == "__main__":
#     # 지구 및 위성 파라미터 설정
#     mu = 398600.4418        # 지구의 중력상수 (km^3/s^2)
#     R_earth = 6371.0        # 지구 반지름 (km)
#     # altitude = 500.0        # 위성 고도 (km)
#     altitude = 570
#     a = R_earth + altitude  # 반장축 (km)
    
#     # 거의 원형 궤도 (저궤도 위성)
#     # e = 0.001               # 이심률
#     e = 0.0
#     tau = 0.0               # t=0에서 근일점 통과
#     # 각도들은 deg 단위로 입력
#     Omega = 0.0             # 승교점 적경 (deg)
#     w = 0.0                 # 근일점 인자 (deg)
#     # i = 51.6                # 경사각 (deg), 예를 들어 ISS와 유사
#     i = 70
    
#     # 케플러의 제3법칙을 이용하여 궤도 주기 계산 (초 단위)
#     T = 2 * np.pi * np.sqrt(a**3 / mu)
    
#     # 궤도 전파
#     times, coords = propagate_leo(a, T, e, tau, Omega, w, i, num_points=200)
    
#     # x-y 평면 상에 궤적 플로팅 (좌표 단위: km)
#     plt.figure(figsize=(8, 8))
#     plt.plot(coords[:, 0], coords[:, 1], '-o', markersize=3, label="Orbit Trajectory")
#     plt.xlabel("X coordinate (km)")
#     plt.ylabel("Y coordinate (km)")
#     plt.title("LEO Satellite Orbit Projection (X-Y plane)")
#     plt.axis("equal")
#     plt.grid(True)
#     plt.legend()
#     plt.show()
    
#     # 궤도 주기 및 반장축 출력
#     print("Orbital period T (s):", T)
#     print("Semi-major axis a (km):", a)

#     # 시뮬레이션 결과 좌표 출력 (선택 사항)
#     for t, pos in zip(times, coords):
#         print("t = {:.2f} sec: x = {:.2f} km, y = {:.2f} km, z = {:.2f} km".format(t, pos[0], pos[1], pos[2]))
# if __name__ == "__main__":
#     # 예시: Walker(720/36/1) 구성
#     ke_list = generate_walker_delta_constellation()
    
#     print(f"총 생성된 위성 수: {len(ke_list)}")
#     print("출력:")
#     # for idx, ke in enumerate(ke_list[:5]):
#     for idx, ke in enumerate(ke_list):
#         a, e, i, RAAN, w, M = ke
#         print(f" Satellite {idx+1}: a={a:.2f} km, e={e}, i={i:.2f} deg, RAAN={RAAN:.2f} deg, w={w:.2f} deg, M={M:.2f} deg")

if __name__ == "__main__":
    # 예시: Walker(720/36/1), 0~6000초 구간을 10개 시점으로 전파
    results = propagate_walker_delta_constellation(
        total_sat=20,
        num_planes=5,
        altitude_km=570.0,
        inclination_deg=70.0,
        F=1,
        t_end=6000.0,
        num_points=100
    )
    
    # 결과가 매우 많으므로, 일부분만 출력
    print(f"총 {len(results)}개 위성 전파 결과")
    
    # for r in results:
    #     sat_id = r["sat_id"]
    #     kepler = r["kepler"]  # (a, e, i, RAAN, w, M)
    #     times = r["times"]
    #     coords = r["coords"]
        
    #     print(f"\n위성 ID = {sat_id}, 케플러 요소 = {kepler}")
    #     for t, (x, y, z) in zip(times, coords):
    #         print(f"  t={t:.1f}s => x={x:.2f} km, y={y:.2f} km, z={z:.2f} km")
    plot_constellation_3d(results)
    save_constellation_csv(results)