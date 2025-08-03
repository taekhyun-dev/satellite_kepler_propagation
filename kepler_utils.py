import numpy as np
import datetime
from typing import List, Tuple


def kepler_solver(M: float, e: float, tol: float = 1e-8, max_iter: int = 100) -> float:
    """Solve Kepler's equation for eccentric anomaly."""
    if e < 0.8:
        E = M
    else:
        E = np.pi
    for _ in range(max_iter):
        f = E - e * np.sin(E) - M
        f_prime = 1 - e * np.cos(E)
        E_new = E - f / f_prime
        if np.abs(E_new - E) < tol:
            return float(E_new)
        E = E_new
    raise RuntimeError("Kepler solver did not converge")


def generate_walker_delta_constellation(
    total_sat: int = 20,
    num_planes: int = 5,
    altitude_km: float = 570.0,
    inclination_deg: float = 70.0,
    earth_radius_km: float = 6371.0,
    e: float = 0.0,
    arg_perigee_deg: float = 0.0,
    F: int = 1,
    Omega_0: float = 0.0,
    M_0: float = 0.0,
) -> List[Tuple[float, float, float, float, float, float]]:
    """Generate Walker Delta constellation kepler elements."""
    a = earth_radius_km + altitude_km
    sats_per_plane = total_sat // num_planes
    delta_RAAN = 360.0 / num_planes
    delta_M = 360.0 / sats_per_plane
    elements = []
    for p in range(num_planes):
        for s in range(sats_per_plane):
            RAAN_p = Omega_0 + p * delta_RAAN
            M_ps = (
                M_0
                + s * delta_M
                + p * (F * 360.0) / (num_planes * sats_per_plane)
            )
            elements.append((a, e, inclination_deg, RAAN_p, arg_perigee_deg, M_ps))
    return elements


def generate_tle_from_kepler(
    a: float,
    e: float,
    i_deg: float,
    RAAN_deg: float,
    w_deg: float,
    M_deg: float,
    epoch_dt: datetime.datetime,
    satnum: int,
    mu: float = 398600.4418,
) -> Tuple[str, str]:
    """Create TLE lines from keplerian elements."""
    import math

    T = 2 * math.pi * math.sqrt(a ** 3 / mu)
    mean_motion = 86400.0 / T
    epoch_year = epoch_dt.year % 100
    day_of_year = (epoch_dt - datetime.datetime(epoch_dt.year, 1, 1)).total_seconds() / 86400.0 + 1
    epoch_str = f"{epoch_year:02d}{day_of_year:012.8f}"
    line1 = f"1 {satnum:05d}U 20000A   {epoch_str}  .00000000  00000-0  00000-00  9990"
    line1 = line1.ljust(69)
    ecc_str = f"{e:.7f}"[2:]
    line2 = (
        f"2 {satnum:05d} {i_deg:8.4f} {RAAN_deg:8.4f} {ecc_str:7s} "
        f"{w_deg:8.4f} {M_deg:8.4f} {mean_motion:11.8f}00000"
    )
    line2 = line2.ljust(68) + "0"
    return line1, line2
