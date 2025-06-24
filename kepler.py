import numpy as np

def kepler_solver(M, e, tol=1e-8, max_iter=100):
    """
    평균 anomaly M과 이심률 e를 입력받아 eccentric anomaly E를 계산하는 함수.
    뉴턴-랩슨 방법을 사용합니다.
    """
    # 초기 추정값: e가 작으면 M을, 그렇지 않으면 pi를 사용합니다.
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

def propagate_kepler(a, e, i, RAAN, arg_periapsis, M0, t0, t, mu=398600.4418):
    """
    궤도 요소와 시간 정보를 기반으로 궤도 전파를 수행합니다.
    
    매개변수:
    - a: 반장축 (km)
    - e: 이심률
    - i: 경사각 (radians)
    - RAAN: 승교점 적경 (radians)
    - arg_periapsis: 근일점 인자 (radians)
    - M0: 초기 평균 anomaly (radians)
    - t0: 기준 시각 (초)
    - t: 목표 시각 (초)
    - mu: 중심체의 중력상수 (기본값: 지구, km^3/s^2)
    
    반환값:
    - 관성 좌표계에서의 위치 벡터 (km)
    - 궤도 반지름 r (km)
    - 진위각 (radians)
    """
    # 평균 운동 계산
    n = np.sqrt(mu / a**3)
    # 목표 시각의 평균 anomaly 계산
    M = M0 + n * (t - t0)
    
    # Kepler 방정식을 풀어 eccentric anomaly 계산
    E = kepler_solver(M, e)
    
    # E를 통해 진위각 (true anomaly) 계산
    cos_nu = (np.cos(E) - e) / (1 - e * np.cos(E))
    sin_nu = (np.sqrt(1 - e**2) * np.sin(E)) / (1 - e * np.cos(E))
    nu = np.arctan2(sin_nu, cos_nu)
    
    # 중심체로부터의 거리
    r = a * (1 - e * np.cos(E))
    
    # 궤도평면상의 좌표
    x_orb = r * np.cos(nu)
    y_orb = r * np.sin(nu)
    
    # 궤도평면 좌표를 관성 좌표계로 변환하기 위한 회전 행렬
    cos_RAAN = np.cos(RAAN)
    sin_RAAN = np.sin(RAAN)
    cos_i = np.cos(i)
    sin_i = np.sin(i)
    cos_arg = np.cos(arg_periapsis)
    sin_arg = np.sin(arg_periapsis)
    
    R11 = cos_RAAN * cos_arg - sin_RAAN * sin_arg * cos_i
    R12 = -cos_RAAN * sin_arg - sin_RAAN * cos_arg * cos_i
    R21 = sin_RAAN * cos_arg + cos_RAAN * sin_arg * cos_i
    R22 = -sin_RAAN * sin_arg + cos_RAAN * cos_arg * cos_i
    R31 = sin_arg * sin_i
    R32 = cos_arg * sin_i
    
    # 관성 좌표계에서의 위치
    x = R11 * x_orb + R12 * y_orb
    y = R21 * x_orb + R22 * y_orb
    z = R31 * x_orb + R32 * y_orb
    
    return np.array([x, y, z]), r, nu

# 예제 실행
if __name__ == "__main__":
    # 예제 궤도 요소 설정
    a = 7000          # km (반장축)
    e = 0.1           # 이심률
    i = np.radians(98)  # 경사각 (rad)
    RAAN = np.radians(30)         # 승교점 적경 (rad)
    arg_periapsis = np.radians(45)  # 근일점 인자 (rad)
    M0 = np.radians(0)            # 초기 평균 anomaly (rad)
    t0 = 0          # 기준 시각 (초)
    t = 3600        # 1시간 후 (초)
    
    pos, r, nu = propagate_kepler(a, e, i, RAAN, arg_periapsis, M0, t0, t)
    
    print("관성 좌표계에서의 위치 벡터 (km):", pos)
    print("중심체로부터의 거리 (km):", r)
    print("진위각 (deg):", np.degrees(nu))
