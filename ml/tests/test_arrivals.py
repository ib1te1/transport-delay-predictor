import numpy as np

from busdelay.arrivals import detect_arrivals
from world import make_plan, make_track


def _detect(plan, track, departs=None):
    return detect_arrivals(
        track["t"].to_numpy(),
        track["lat"].to_numpy(),
        track["lon"].to_numpy(),
        plan["t_plan"].to_numpy(),
        plan["lat"].to_numpy(),
        plan["lon"].to_numpy(),
        departs=departs,
    )


def test_arrivals_match_the_true_delays():
    plan = make_plan(n_stops=20)
    delays = np.linspace(0, 150, 20)
    arrived, dwell, dist = _detect(plan, make_track(plan, delays, step_s=5))
    error = arrived - (plan["t_plan"].to_numpy() + delays)
    # the toy bus already stands at the first stop when the track begins, so the closest
    # fix there is the start of the track; the rest must be exact up to the fix step
    assert np.abs(error[1:]).max() <= 5
    assert (dwell >= 15).all()
    assert (dist < 20).all()


def test_departure_from_the_first_stop_of_a_trip():
    plan = make_plan(n_stops=5)
    delays = np.full(5, 30.0)
    departs = np.array([True, False, False, False, False])
    arrived, _, _ = _detect(plan, make_track(plan, delays, dwell_s=20, step_s=5), departs)
    # leaves the first stop 20 s after "arriving" there
    assert arrived[0] - (plan["t_plan"].iloc[0] + 30) == 20
    assert abs(arrived[1] - (plan["t_plan"].iloc[1] + 30)) <= 5


def test_stops_not_reached_yet_are_empty():
    plan = make_plan(n_stops=20)
    track = make_track(plan, np.zeros(20))
    cut = track[track["t"] <= plan["t_plan"].iloc[9] + 30]
    arrived, _, _ = _detect(plan, cut)
    assert not np.isnan(arrived[:10]).any()
    assert np.isnan(arrived[11:]).all()


def test_no_telemetry_at_all():
    plan = make_plan(n_stops=5)
    empty = np.array([])
    arrived, dwell, dist = detect_arrivals(
        empty, empty, empty, plan["t_plan"], plan["lat"], plan["lon"]
    )
    assert np.isnan(arrived).all() and np.isnan(dwell).all() and np.isnan(dist).all()


def test_pass_closest_to_the_planned_time_wins():
    # one stop, the bus drives by twice: six minutes before the plan and right on time
    stop_t = np.array([10_000.0])
    lat = np.array([55.75])
    lon = np.array([37.6])
    t = np.array([9_630, 9_640, 9_650, 9_990, 10_000, 10_010], dtype=float)
    far = 37.61
    fix_lon = np.array([37.6, 37.6, far, far, 37.6, 37.6])
    arrived, _, _ = detect_arrivals(t, np.full(6, 55.75), fix_lon, stop_t, lat, lon)
    assert arrived[0] == 10_000
