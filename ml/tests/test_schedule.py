import numpy as np
import pandas as pd

from busdelay.schedule import VehiclePlan, prepare_plan, route_ids
from world import make_plan


def test_trips_are_cut_at_layovers():
    plan = prepare_plan(make_plan(n_stops=10, every_s=60, layovers=[(4, 600)]))
    assert plan["trip"].tolist() == [0] * 5 + [1] * 5
    assert plan["seq"].tolist() == list(range(5)) * 2
    assert plan["trip_len"].tolist() == [5] * 10
    assert plan["gap_s"].iloc[5] == 660
    assert np.isnan(plan["gap_s"].iloc[0])


def test_distances_and_keys():
    plan = prepare_plan(make_plan(n_stops=4))
    assert plan["step_m"].iloc[0] == 0
    assert plan["step_m"].iloc[1:].between(300, 330).all()
    assert plan["cum_m"].is_monotonic_increasing
    assert plan["key"].nunique() == 4


def test_plan_is_sorted_by_time_per_vehicle():
    shuffled = make_plan(tr_id=1, n_stops=5).sample(frac=1, random_state=0)
    plan = prepare_plan(shuffled)
    assert plan["t_plan"].is_monotonic_increasing
    assert plan["pos"].tolist() == [0, 1, 2, 3, 4]


def test_copies_of_a_route_get_one_id():
    real = make_plan(tr_id=5, n_stops=20)
    copy = make_plan(tr_id=9_000_001, n_stops=20, start=real["t_plan"].iloc[0] + 300)
    other = make_plan(tr_id=7, n_stops=20, row=1)
    routes = route_ids(pd.concat([copy, real, other]))
    assert routes == {5: 5, 7: 7, 9_000_001: 5}


def test_last_planned_and_target_window():
    frame = prepare_plan(make_plan(n_stops=30, start=1000.0, every_s=60))
    plan = VehiclePlan.from_frame(frame)
    assert plan.last_planned(999) == -1
    assert plan.last_planned(1000) == 0
    assert plan.last_planned(1119) == 1
    # T = 1000: window (1600, 1900] -> first stop is planned at 1660, index 11
    assert plan.pick_target(1000) == 11
    assert plan.pick_target(1000 + 30 * 60) == -1
