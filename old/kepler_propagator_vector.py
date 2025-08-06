import numpy as np
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
    지정된 시간 범위에서 궤도 좌표 (x, y, z)를 계산해 반환합니다.
    
    매개변수:
    -----------
    a : float
        반장축 (예: AU, km 등)
    per : float
        공전 주기 (a와 동일 단위를 가정. 예: 일(day))
    e : float
        이심률(0~1)
    tau : float
        근일점 통과 시점(단위: per와 동일).
        예: tau=0 이면 t=0에서 근일점 통과
    Omega : float
        승교점 적경 (deg)
    w : float
        근일점 인자 (deg)
    i : float
        궤도 경사각 (deg)
    start_time : float
        궤도 전파 시작 시각 (기본값 0.0, per와 동일 단위)
    end_time : float or None
        궤도 전파 종료 시각 (None이면 per로 설정)
    num_points : int
        start_time ~ end_time 구간을 몇 등분하여 계산할지 결정 (기본 100)
        
    반환값:
    -----------
    times : numpy.ndarray
        전파 시각 배열 (start_time ~ end_time까지)
    coords : numpy.ndarray
        (num_points, 3) 형태의 2차원 배열. 각 행은 [x, y, z].
        단위는 a와 동일 (예: a가 AU이면 AU, km이면 km).
    """
    # end_time이 지정되지 않았다면, 한 주기를 기본으로 설정
    if end_time is None:
        end_time = per
    
    # KeplerEllipse 객체 생성
    # tau: 근일점 통과 시점
    ke = pyasl.KeplerEllipse(a, per, e=e, tau=tau, Omega=Omega, w=w, i=i)
    
    # 전파할 시간 배열 생성 (start_time ~ end_time, num_points개)
    times = np.linspace(start_time, end_time, num_points)
    
    # 결과를 저장할 배열 (num_points x 3)
    coords = np.zeros((num_points, 3), dtype=float)
    
    # 각 시점에서 (x, y, z) 계산
    for idx, t in enumerate(times):
        x, y, z = ke.xyzPos(t)
        coords[idx, :] = [x, y, z]
    
    return times, coords


if __name__ == "__main__":
    # 예시 케플러 요소
    a_example = 1.0         # 반장축 (예: AU)
    per_example = 365.25    # 공전 주기 (예: 일)
    e_example = 0.0167      # 이심률
    tau_example = 0.0       # 근일점 통과 시점 (per와 동일 단위)
    Omega_example = 0.0     # 승교점 적경 (deg)
    w_example = 102.9       # 근일점 인자 (deg)
    i_example = 0.0         # 경사각 (deg)
    
    # 0일부터 365일까지, 10개 지점에 대해 궤도 좌표 계산
    times_out, coords_out = propagate_kepler_pyastronomy(
        a_example, per_example, e=e_example, tau=tau_example,
        Omega=Omega_example, w=w_example, i=i_example,
        start_time=0.0, end_time=365.0, num_points=10
    )
    
    # 결과 출력
    print("Time array (days):", times_out)
    print("Coordinates (x, y, z) [same unit as 'a']:")
    for t, (x, y, z) in zip(times_out, coords_out):
        print(f" t = {t:8.2f} -> x = {x:10.6f}, y = {y:10.6f}, z = {z:10.6f}")