# simserver/sim/sky.py
from skyfield.api import load, EarthSatellite, Topos
from datetime import datetime

ts = load.timescale()

def to_ts(dt: datetime):
    return ts.utc(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second)

def get_current_time_utc(sim_time):
    return to_ts(sim_time)

def elevation_deg(sat: EarthSatellite, topox: Topos, t_ts):
    alt, _, _ = (sat - topox).at(t_ts).altaz()
    return alt.degrees
