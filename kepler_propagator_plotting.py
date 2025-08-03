import numpy as np
import matplotlib.pyplot as plt
from PyAstronomy import pyasl

from kepler_utils import kepler_solver

def kepler_to_state(a, e, inc, RAAN, arg, M, mu):
    """
    케플러 요소로부터 관성 좌표계의 위치 및 속도 벡터를 계산합니다.

    매개변수:
      - a: 반장축 (km)
      - e: 이심률
      - inc: 경사각 (radians)
      - RAAN: 승교점 적경 (radians)
      - arg: 근일점 인자 (radians)
      - M: 평균 anomaly (radians)
      - mu: 중심체의 중력상수 (km^3/s^2, 예: 지구의 mu=398600.4418)
    
    반환값:
      - r_inertial: 관성 좌표계에서의 위치 벡터 (km)
      - v_inertial: 관성 좌표계에서의 속도 벡터 (km/s)
    """
    # 1. 평균 anomaly로부터 eccentric anomaly 계산
    E = kepler_solver(M, e)

    # 2. eccentric anomaly를 이용해 진위각(ν) 계산
    #    tan(ν/2) = sqrt((1+e)/(1-e)) * tan(E/2)
    nu = 2.0 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2.0), np.sqrt(1 - e) * np.cos(E / 2.0))

    # 3. 중심체로부터의 거리 및 궤도 이심률에 따른 반력 계산
    r = a * (1 - e * np.cos(E))
    p = a * (1 - e**2)

    # 4. 궤도평면(perifocal frame)에서의 위치 및 속도 계산
    # 위치: [r*cos(ν), r*sin(ν), 0]
    r_pf = np.array([r * np.cos(nu), r * np.sin(nu), 0.0])
    # 속도: sqrt(mu/p) * [ -sin(ν), e + cos(ν), 0 ]
    v_pf = np.array([-np.sqrt(mu / p) * np.sin(nu),
                      np.sqrt(mu / p) * (e + np.cos(nu)),
                      0.0])

    # 5. 관성 좌표계로 변환 (Perifocal -> Inertial)
    # 회전 행렬:  R = R3(-RAAN) * R1(-inc) * R3(-arg)
    # 다만, 여기서는 아래와 같이 직접 구성합니다.
    cos_RAAN = np.cos(RAAN)
    sin_RAAN = np.sin(RAAN)
    cos_arg = np.cos(arg)
    sin_arg = np.sin(arg)
    cos_inc = np.cos(inc)
    sin_inc = np.sin(inc)

    R = np.array([
        [cos_RAAN * cos_arg - sin_RAAN * sin_arg * cos_inc, -cos_RAAN * sin_arg - sin_RAAN * cos_arg * cos_inc, sin_RAAN * sin_inc],
        [sin_RAAN * cos_arg + cos_RAAN * sin_arg * cos_inc, -sin_RAAN * sin_arg + cos_RAAN * cos_arg * cos_inc, -cos_RAAN * sin_inc],
        [sin_arg * sin_inc,                                cos_arg * sin_inc,                                 cos_inc]
    ])

    # 6. 관성 좌표계에서의 위치 및 속도 벡터 계산
    r_inertial = R.dot(r_pf)
    v_inertial = R.dot(v_pf)

    return r_inertial, v_inertial

def propagate_kepler_pyastronomy(a, per, e=0.0, tau=0.0, Omega=0.0, w=0.0, i=0.0,
                                 start_time=0.0, end_time=None, num_points=100):
    """
    PyAstronomy의 KeplerEllipse를 사용하여,
    지정된 시간 범위 내의 궤도 좌표 (x, y, z)를 계산합니다.
    
    매개변수:
      a         : 반장축 (예: AU, km 등)
      per       : 공전 주기 (start_time, end_time과 동일한 시간 단위; 예: 일(day))
      e         : 이심률 (0~1)
      tau       : 근일점 통과 시점 (per와 동일 단위; 예: tau=0이면 t=0에서 근일점)
      Omega     : 승교점 적경 (deg)
      w         : 근일점 인자 (deg)
      i         : 궤도 경사각 (deg)
      start_time: 전파 시작 시각 (per와 동일 단위)
      end_time  : 전파 종료 시각 (None이면 per로 설정)
      num_points: 계산할 시점의 개수
      
    반환값:
      times  : 시각 배열
      coords : 각 시각에서의 [x, y, z] 좌표 (2차원 배열, shape=(num_points, 3))
    """
    if end_time is None:
        end_time = per

    # KeplerEllipse 객체 생성 (Tref는 지원되지 않으므로, tau로 근일점 통과 시점을 지정)
    ke = pyasl.KeplerEllipse(a, per, e=e, tau=tau, Omega=Omega, w=w, i=i)
    
    # 시간 배열 생성
    times = np.linspace(start_time, end_time, num_points)
    coords = np.zeros((num_points, 3), dtype=float)
    
    for idx, t in enumerate(times):
        x, y, z = ke.xyzPos(t)
        coords[idx, :] = [x, y, z]
    
    return times, coords

def plot_orbit(a, per, e=0.0, tau=0.0, Omega=0.0, w=0.0, i=0.0,
               start_time=0.0, end_time=None, num_points=200):
    """
    propagate_kepler_pyastronomy()로 계산한 궤도 좌표를 x-y 평면 상에 플로팅합니다.
    """
    times, coords = propagate_kepler_pyastronomy(a, per, e, tau, Omega, w, i, start_time, end_time, num_points)
    x = coords[:, 0]
    y = coords[:, 1]
    
    plt.figure(figsize=(8, 8))
    plt.plot(x, y, '-o', markersize=3, label="Orbit Trajectory")
    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    plt.title("Kepler Orbit Propagation")
    plt.grid(True)
    plt.legend()
    plt.axis('equal')  # 축의 비율을 동일하게 하여 궤도의 형태를 정확히 표현
    plt.show()

if __name__ == "__main__":
    # 예시 케플러 요소 (예: 태양 주위를 도는 행성)
    a_example = 1.0         # 반장축 (단위: AU)
    per_example = 365.25    # 공전 주기 (단위: 일)
    e_example = 0.0167      # 이심률
    tau_example = 0.0       # 근일점 통과 시점 (t=0에서 근일점)
    Omega_example = 0.0     # 승교점 적경 (deg)
    w_example = 102.9       # 근일점 인자 (deg)
    i_example = 0.0         # 경사각 (deg)
    
    # 전체 좌표 계산 및 플로팅
    plot_orbit(a_example, per_example, e=e_example, tau=tau_example,
               Omega=Omega_example, w=w_example, i=i_example,
               start_time=0.0, end_time=per_example, num_points=200)
