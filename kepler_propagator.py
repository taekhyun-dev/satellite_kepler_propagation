import numpy as np
import matplotlib.pyplot as plt
from PyAstronomy import pyasl

def kepler_solver(M, e, tol=1e-8, max_iter=100):
    """
    평균 anomaly M과 이심률 e를 입력받아 eccentric anomaly E를 계산하는 함수.
    뉴턴-랩슨(Newton-Raphson) 방법을 사용합니다.
    """
    if e < 0.8:
        E = M
    else:
        E = np.pi

    for _ in range(max_iter):
        f = E - e * np.sin(E) - M
        f_prime = 1 - e * np.cos(E)
        E_new = E - f / f_prime
        if np.abs(E_new - E) < tol:
            return E_new
        E = E_new

    raise RuntimeError("Kepler solver did not converge")

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

def example_kepler_propagation():
    """
    PyAstronomy의 KeplerEllipse를 사용하여
    시간에 따른 케플러 궤도 전파를 시현하는 예시 함수
    """
    # 예시 케플러 요소 (단위 임의, 여기서는 태양-행성계처럼 가정)
    a = 1.0         # 반장축 (AU 등)
    period = 365.25 # 공전 주기 (days)
    e = 0.0167      # 이심률
    Omega = 0.0     # 승교점 적경 (deg) - 편의상 0
    w = 102.9       # 근일점 인자(deg)
    i = 0.0         # 경사각(deg)
    tau = 0.0       # 근일점 통과 시점(임의)
    Tref = 2451545.0 # 기준 epoch (JD)

    # KeplerEllipse 생성
    ke = pyasl.KeplerEllipse(a, period, e=e, Omega=Omega, w=w, i=i, tau=tau)
    # ke = pyasl.KeplerEllipse(a, period, e=e, Omega=Omega, w=w, i=i, tau=tau, Tref=Tref)

    # 0 ~ 1공전 주기를 100분할하여 위치 계산
    t_array = np.linspace(0, period, 100)  # 일 단위
    xs, ys, zs = [], [], []

    for t in t_array:
        # xyzPos(t): 주어진 t(일)에서의 x, y, z (기본 단위: a와 동일)
        x, y, z = ke.xyzPos(t)
        xs.append(x)
        ys.append(y)
        zs.append(z)

    # 2D 플롯
    plt.figure(figsize=(6, 6))
    plt.plot(xs, ys, 'b.-', label="Orbit")
    plt.plot(xs[0], ys[0], 'ro', label="Start")
    plt.gca().set_aspect('equal', 'box')
    plt.xlabel("x [AU]")
    plt.ylabel("y [AU]")
    plt.title("KeplerEllipse Orbit Propagation Example")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # # 예제 케플러 요소 (단위: km, radian)
    # a = 7000.0                           # 반장축 (km)
    # e = 0.1                              # 이심률
    # inc = np.radians(98.0)               # 경사각 (radians)
    # RAAN = np.radians(30.0)              # 승교점 적경 (radians)
    # arg = np.radians(45.0)               # 근일점 인자 (radians)
    # M = 0.0                            # 평균 anomaly (radians)
    # mu = 398600.4418                     # 지구의 중력상수 (km^3/s^2)

    # r_vec, v_vec = kepler_to_state(a, e, inc, RAAN, arg, M, mu)

    # print("관성 좌표계 위치 벡터 (km):", r_vec)
    # print("관성 좌표계 속도 벡터 (km/s):", v_vec)

    example_kepler_propagation()