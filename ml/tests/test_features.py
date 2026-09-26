import numpy as np
import pandas as pd
import pytest

from busdelay.features import FEATURES, Track, build_features, point_features
from busdelay.schedule import VehiclePlan, prepare_plan
from world import make_plan, make_track


def _setup(delays=None, layovers=()):
    plan = make_plan(n_stops=40, every_s=60, layovers=layovers)
    if delays is None:
        delays = np.full(len(plan), 90.0)
    track = make_track(plan, delays)
    return plan, track


def _point(plan, T, cur_dev_s=90.0):
    vehicle = VehiclePlan.from_frame(prepare_plan(plan))
    target = vehicle.pick_target(T)
    return pd.DataFrame(
        {
            "sample_id": ["p"],
            "tr_id": [1],
            "T": [T],
            "target_stop_id": [vehicle.stop_id[target]],
            "t_target": [vehicle.t_plan[target]],
            "cur_dev_s": [cur_dev_s],
            "target": [np.nan],
            "part": ["validate"],
        }
    )


def test_every_feature_is_present_and_in_order():
    plan, track = _setup()
    T = plan["t_plan"].iloc[15]
    table = build_features(_point(plan, T), plan, track)
    assert list(table.columns[-len(FEATURES) :]) == FEATURES
    assert len(table) == 1


def test_telemetry_after_T_changes_nothing():
    plan, track = _setup()
    T = plan["t_plan"].iloc[15] + 17
    points = _point(plan, T)
    honest = build_features(points, plan, track)

    future = track["t"] > T
    spoiled = track.copy()
    spoiled.loc[future, "lat"] += 0.3
    spoiled.loc[future, "speed"] = 99.0
    spoiled.loc[future, "t"] += 5.0
    assert future.any()
    pd.testing.assert_frame_equal(honest, build_features(points, plan, spoiled))


def test_point_features_refuses_a_track_from_the_future():
    plan, track = _setup()
    vehicle = VehiclePlan.from_frame(prepare_plan(plan))
    T = plan["t_plan"].iloc[15]
    with pytest.raises(ValueError, match="after T"):
        point_features(vehicle, Track.from_frame(track), T, vehicle.stop_id[27], 0.0)


def test_plan_with_facts_is_refused():
    plan, track = _setup()
    T = plan["t_plan"].iloc[15]
    with pytest.raises(ValueError, match="fact"):
        build_features(_point(plan, T), plan.assign(t_fact=0.0), track)


def test_gps_delay_and_timetable_features():
    plan, track = _setup()
    T = plan["t_plan"].iloc[15] + 30
    row = build_features(_point(plan, T), plan, track).iloc[0]
    assert row["gps_dev_last_s"] == pytest.approx(90, abs=15)
    assert row["gps_dev_mean3_s"] == pytest.approx(90, abs=15)
    assert row["lead_s"] == 630  # target is stop 26, 11 minutes after stop 15, T is 30 s later
    assert row["stops_ahead"] == 11
    assert row["new_trip_ahead"] == 0
    assert row["layover_ahead_s"] == 0
    assert row["tel_age_s"] <= 10
    assert row["speed_mean_15m"] > 0


def test_layover_before_the_target():
    plan, track = _setup(layovers=[(19, 480)])
    T = plan["t_plan"].iloc[17]
    row = build_features(_point(plan, T, cur_dev_s=200.0), plan, track).iloc[0]
    assert row["new_trip_ahead"] == 1
    assert row["layover_ahead_s"] == 540
    assert row["layover_slack_s"] == 340


def test_gps_hint_replaces_the_given_one():
    plan, track = _setup()
    T = plan["t_plan"].iloc[15] + 30
    points = _point(plan, T, cur_dev_s=-999.0)
    row = build_features(points, plan, track, hint="gps").iloc[0]
    assert row["cur_dev_s"] == pytest.approx(90, abs=15)
    with pytest.raises(ValueError, match="hint"):
        build_features(points, plan, track, hint="future")


def test_vehicle_without_telemetry_gets_nan_not_an_error():
    plan, track = _setup()
    T = plan["t_plan"].iloc[15]
    row = build_features(_point(plan, T), plan, track.iloc[0:0]).iloc[0]
    assert np.isnan(row["gps_dev_last_s"])
    assert np.isnan(row["tel_age_s"])
    assert row["cur_dev_s"] == 90


def test_lose_drops_or_blanks_the_fixes_after_the_moment():
    plan, track = _setup()
    full = Track.from_frame(track)
    after = plan["t_plan"].iloc[10]

    silent = full.lose(after)
    assert silent.t.max() <= after and len(silent) == len(full.upto(after))

    blind = full.lose(after, keep_fixes=True)
    lost = full.t > after
    assert len(blind) == len(full)
    assert not blind.ok[lost].any() and np.isnan(blind.lat[lost]).all()
    np.testing.assert_array_equal(blind.lat[~lost], full.lat[~lost])
    assert full.ok.all()


@pytest.mark.parametrize("kind", ["silence", "no_position"])
def test_outage_columns_cut_the_track_before_T(kind):
    plan, track = _setup()
    T = plan["t_plan"].iloc[15] + 17
    points = _point(plan, T).assign(gap_s=300.0, outage=kind)
    cut = build_features(points, plan, track, hint="gps")
    assert list(cut.columns[:9]) == [
        "sample_id", "tr_id", "part", "T", "target", "synthetic", "route", "gap_s", "outage"
    ]  # fmt: skip
    assert list(cut.columns[9:]) == FEATURES

    row = cut.iloc[0]
    assert row["tel_age_s"] >= 300
    if kind == "silence":
        by_hand = build_features(_point(plan, T), plan, track[track["t"] <= T - 300], hint="gps")
        pd.testing.assert_series_equal(row[FEATURES], by_hand.iloc[0][FEATURES], check_names=False)
    else:
        # the fixes still come, only without a position
        fresh = build_features(_point(plan, T), plan, track, hint="gps").iloc[0]
        assert row["fixes_15m"] == fresh["fixes_15m"] and row["ok_share_15m"] < 1
