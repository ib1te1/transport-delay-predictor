import numpy as np
import pandas as pd
import pytest

from busdelay.data import load_part
from busdelay.explain import REASONS
from busdelay.features import build_features
from busdelay.inference import Forecaster, queries_for_points, top_reasons
from busdelay.model import TrainConfig, train
from world import write_dataset

SMALL = {"iterations": 40, "depth": 3, "learning_rate": 0.2, "folds": 3}


@pytest.fixture(scope="module")
def parts(tmp_path_factory):
    data = write_dataset(tmp_path_factory.mktemp("raw"))
    return {name: load_part(data, name) for name in ("train", "test")}


@pytest.fixture(scope="module")
def model(parts):
    table = pd.concat(
        [build_features(p.points, p.plan, p.telemetry) for p in parts.values()],
        ignore_index=True,
    )
    return train(table, TrainConfig(**SMALL))[0]


@pytest.mark.parametrize("hint", ["gps", "given"])
def test_service_path_gives_the_offline_predictions(parts, model, hint):
    test = parts["test"]
    offline = model.predict(build_features(test.points, test.plan, test.telemetry, hint=hint))
    points = test.points
    if hint == "gps":
        # nothing sent with the point: the estimate is all there is, as offline
        points = points.assign(cur_dev_s=np.nan)
    queries = queries_for_points(points, test.plan, test.telemetry)

    answers = Forecaster(model, hint=hint).predict(queries)

    assert [a.sample_id for a in answers] == points["sample_id"].tolist()
    assert all(a.fallback is None for a in answers)
    np.testing.assert_allclose([a.delay_s for a in answers], offline["delay_s"], atol=1e-9)
    np.testing.assert_allclose([a.late_prob for a in answers], offline["late_prob"], atol=1e-9)


def test_the_sent_hint_fills_in_when_the_gps_estimate_fails(parts, model):
    test = parts["test"]
    queries = queries_for_points(test.points.iloc[:2], test.plan, test.telemetry)
    for q in queries:
        q.fixes = q.fixes.iloc[:0]
        q.cur_dev_s = 42.0
    answers = Forecaster(model).predict(queries)
    assert [a.cur_dev_s for a in answers] == [42.0, 42.0]
    assert all(np.isfinite(a.delay_s) for a in answers)


def test_a_broken_point_falls_back_without_failing_the_batch(parts, model):
    test = parts["test"]
    queries = queries_for_points(test.points.iloc[:3], test.plan, test.telemetry)
    queries[1].target_stop_id = -1
    queries[1].cur_dev_s = 75.0

    answers = Forecaster(model).predict(queries)

    assert [a.sample_id for a in answers] == [q.sample_id for q in queries]
    assert answers[1].fallback is not None and answers[1].delay_s == 75.0
    assert answers[1].reasons == [] and np.isnan(answers[1].late_prob)
    assert answers[0].fallback is None and answers[2].fallback is None


def test_answers_tell_how_old_the_last_position_is(parts, model):
    test = parts["test"]
    queries = queries_for_points(test.points.iloc[:3], test.plan, test.telemetry)
    fresh, silent, blind = queries
    silent.fixes = silent.fixes[silent.fixes["t"] <= silent.T - 600]
    blind.fixes = blind.fixes.assign(location_valid=False)

    answers = Forecaster(model).predict(queries)

    assert all(a.fallback is None and np.isfinite(a.delay_s) for a in answers)
    assert answers[0].position_age_s <= 10
    assert answers[1].position_age_s >= 600
    assert np.isnan(answers[2].position_age_s)


def test_unknown_hint_is_refused(model):
    with pytest.raises(ValueError, match="hint"):
        Forecaster(model, hint="matcher")


def test_top_reasons_keep_the_big_pushes_towards_late():
    contribution = np.zeros(len(REASONS))
    keys = list(REASONS)
    contribution[keys.index("long_dwell")] = 40.0
    contribution[keys.index("carried_delay")] = 90.0
    contribution[keys.index("layover")] = -200.0
    contribution[keys.index("signal")] = 10.0
    assert top_reasons(contribution) == ["carried_delay", "long_dwell"]
    assert top_reasons(np.zeros(len(REASONS))) == []
