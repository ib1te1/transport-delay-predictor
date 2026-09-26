import numpy as np
import pandas as pd
import pytest

from busdelay.data import load_part
from busdelay.features import Track, build_features, estimate_cur_dev
from busdelay.model import TrainConfig, train
from busdelay.online import LivePredictor, benchmark, risk_level
from busdelay.schedule import VehiclePlan, prepare_plan
from world import make_plan, make_track, write_dataset

SMALL = {"iterations": 30, "depth": 3, "learning_rate": 0.2, "folds": 3}


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    data = write_dataset(tmp_path_factory.mktemp("raw"))
    parts = [load_part(data, name) for name in ("train", "test")]
    table = pd.concat(
        [build_features(p.points, p.plan, p.telemetry) for p in parts], ignore_index=True
    )
    return train(table, TrainConfig(**SMALL))[0]


def _stream(predictor, track, until):
    for fix in track[track["t"] <= until].itertuples():
        predictor.add_fix(fix.tr_id, fix.t, fix.lat, fix.lon, fix.speed, fix.ok)


def test_cur_dev_estimate_follows_the_bus():
    plan = make_plan(n_stops=30)
    track = make_track(plan, np.full(30, 120.0))
    vehicle = VehiclePlan.from_frame(prepare_plan(plan))
    T = plan["t_plan"].iloc[12] + 30
    # stop 12 is planned 30 s ago, the bus is 2 minutes late and has not reached it:
    # the delay seen at stop 10 is used
    estimate = estimate_cur_dev(vehicle, Track.from_frame(track).upto(T), T)
    assert estimate == pytest.approx(120, abs=15)


def test_forecast_on_a_stream(model):
    plan = make_plan(tr_id=1, n_stops=40)
    track = make_track(plan, np.linspace(0, 200, 40))
    predictor = LivePredictor(model, plan)
    now = plan["t_plan"].iloc[15]
    _stream(predictor, track, now)

    forecast = predictor.forecast(1, now)
    assert forecast is not None
    assert 600 < forecast.target_planned - now <= 900
    assert forecast.delay_lo_s <= forecast.delay_s <= forecast.delay_hi_s
    assert 0 <= forecast.late_prob <= 1
    assert forecast.risk in ("green", "yellow", "red")
    assert forecast.reason_text
    assert forecast.degraded is None

    # the stream stops: forecasts go on, marked as degraded
    later = predictor.forecast(1, now + 10 * 60)
    assert later is not None and later.degraded == "no_signal"


def test_one_call_for_many_vehicles_matches_one_by_one(model):
    predictor = LivePredictor(model, pd.concat([make_plan(1, 40), make_plan(2, 40, row=1)]))
    for tr_id, row in ((1, 0), (2, 1)):
        plan = make_plan(tr_id, 40, row=row)
        _stream(predictor, make_track(plan, np.linspace(0, 150, 40)), plan["t_plan"].iloc[15])
    now = make_plan(1, 40)["t_plan"].iloc[15]
    together = predictor.forecast_all(now)
    alone = [predictor.forecast(tr_id, now) for tr_id in (1, 2)]
    assert [f.tr_id for f in together] == [1, 2]
    for a, b in zip(together, alone, strict=True):
        assert a.delay_s == pytest.approx(b.delay_s)
        assert a.reason == b.reason


def test_unknown_vehicle_and_end_of_day(model):
    plan = make_plan(tr_id=1, n_stops=20)
    predictor = LivePredictor(model, plan)
    assert predictor.forecast(999, plan["t_plan"].iloc[0]) is None
    assert predictor.forecast(1, plan["t_plan"].iloc[-1]) is None


def test_old_fixes_leave_the_buffer(model):
    plan = make_plan(tr_id=1, n_stops=40)
    predictor = LivePredictor(model, plan, history_s=600)
    _stream(predictor, make_track(plan, np.zeros(40)), plan["t_plan"].iloc[-1])
    kept = predictor.track(1, plan["t_plan"].iloc[-1] + 60).t
    assert kept[-1] - kept[0] <= 600


def test_benchmark_and_risk_levels(model):
    plan = make_plan(tr_id=1, n_stops=40)
    predictor = LivePredictor(model, plan)
    _stream(predictor, make_track(plan, np.zeros(40)), plan["t_plan"].iloc[20])
    result = benchmark(predictor, [plan["t_plan"].iloc[20]])
    assert result["forecasts"] == 1 and result["cycle_p50_ms"] > 0
    assert [risk_level(p) for p in (0.1, 0.4, 0.9)] == ["green", "yellow", "red"]
