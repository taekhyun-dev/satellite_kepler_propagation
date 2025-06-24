import numpy as np
import matplotlib.pyplot as plt
from PyAstronomy import pyasl

def propagate_geostationary(a, per, num_points=200):
    """
    PyAstronomy의 KeplerEllipse를 사용하여 정지궤도 위성의 궤도 좌표를 계산합니다.
    
    매개변수:
      a         : 반장축 (km) - 지구 반지름 + 위성 고도
      per       : 궤도 주기 (초) - 약 86,164초
      num_points: 계산할 시점의 개수
      
    반환값:
      times : 전파 시각 배열 (초)
      coords: 각 시각에서의 [x, y, z] 좌표 (km 단위, shape=(num_points, 3))
    """
    # 정지궤도는 원형이며, 적도 궤도이므로 e=0, tau=0, Omega=0, w=0, i=0로 설정합니다.
    ke = pyasl.KeplerEllipse(a, per, e=0, tau=0, Omega=0, w=0, i=0)
    
    # 0부터 per까지 num_points개의 시점을 생성
    times = np.linspace(0, per, num_points)
    coords = np.zeros((num_points, 3), dtype=float)
    
    for idx, t in enumerate(times):
        coords[idx, :] = ke.xyzPos(t)
    
    return times, coords

if __name__ == "__main__":
    # 지구 및 정지궤도 위성 파라미터 설정
    R_earth = 6371.0      # 지구 반지름 (km)
    altitude = 35786.0    # 정지궤도 고도 (km)
    a = R_earth + altitude  # 위성 궤도 반장축 (km), 약 42157 km
    
    T = 86164  # 궤도 주기 (초) - 약 한 sidereal day (23시간 56분 4초)
    
    # 궤도 전파
    times, coords = propagate_geostationary(a, T, num_points=200)
    
    # x-y 평면 상의 궤적 플로팅 (단위: km)
    plt.figure(figsize=(8, 8))
    plt.plot(coords[:, 0], coords[:, 1], 'b-', label="Geostationary Orbit")
    plt.xlabel("X (km)")
    plt.ylabel("Y (km)")
    plt.title("Geostationary Orbit Projection (X-Y Plane)")
    plt.axis("equal")  # 축 비율을 동일하게 하여 원형이 왜곡되지 않도록 함
    plt.grid(True)
    plt.legend()
    plt.show()
    
    # 시뮬레이션 결과 좌표 출력 (선택 사항)
    for t, pos in zip(times, coords):
        print("t = {:.2f} sec: x = {:.2f} km, y = {:.2f} km, z = {:.2f} km".format(t, pos[0], pos[1], pos[2]))
        
    # 궤도 주기 및 반장축 출력
    print("Orbital period T (s):", T)
    print("Semi-major axis a (km):", a)