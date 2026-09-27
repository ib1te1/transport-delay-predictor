"""Predictor behind the api: direct contract, scored ticks, current predictions."""

from datetime import timedelta

import pytest
import synthetic
from conftest import API_URL, PREDICTOR_URL, Replayed, Stack, parse_time

REASONS = {
    "accumulated_delay",
    "slow_approach",
    "long_dwell",
    "low_speed_segment",
    "stale_telemetry",
    "unknown",
}
RESPONSE_KEYS = {"sample_id", "prediction_s", "p_late", "reasons", "model_version"}


def _swagger_example(stack: Stack) -> list[dict]:
    """The ready batch predictor puts on its Swagger page for "Try it out"."""
    schema = stack.json(f"{PREDICTOR_URL}/openapi.json")
    body = schema["paths"]["/predict"]["post"]["requestBody"]["content"]["application/json"]
    return body["examples"]["late_bus"]["value"]


def _check_response(answer: dict, sample_id: str) -> None:
    assert set(answer) == RESPONSE_KEYS
    assert answer["sample_id"] == sample_id
    assert isinstance(answer["prediction_s"], float)
    assert answer["p_late"] is None or 0 <= answer["p_late"] <= 1
    assert answer["reasons"] and set(answer["reasons"]) <= REASONS
    assert answer["model_version"]


def test_predictor_answers_a_direct_request(stack: Stack) -> None:
    batch = _swagger_example(stack)
    response = stack.http.post(f"{PREDICTOR_URL}/predict", json=batch)
    assert response.status_code == 200, response.text
    answers = response.json()
    assert len(answers) == 1
    _check_response(answers[0], batch[0]["sample_id"])
    model = stack.json(f"{PREDICTOR_URL}/model")
    assert answers[0]["model_version"] == model["model_version"]


def test_predictor_keeps_batch_order(stack: Stack) -> None:
    request = _swagger_example(stack)[0]
    batch = []
    for n in range(3):
        copy = dict(request, sample_id=f"{request['tr_id']}_{n}")
        batch.append(copy)
    batch.reverse()
    answers = stack.http.post(f"{PREDICTOR_URL}/predict", json=batch).json()
    assert [a["sample_id"] for a in answers] == [r["sample_id"] for r in batch]


def test_predictor_rejects_a_malformed_batch(stack: Stack) -> None:
    response = stack.http.post(f"{PREDICTOR_URL}/predict", json=[{"sample_id": "x"}])
    assert response.status_code == 422


def _predictions(stack: Stack) -> list[dict]:
    columns = (
        "sample_id, tr_id, t, target_stop_id, target_time_begin, prediction_s,"
        " risk_level, degraded, degraded_reason, model_version"
    )
    rows = stack.compose.sql(f"SELECT {columns} FROM predictions ORDER BY t, tr_id")
    names = [c.strip() for c in columns.split(",")]
    return [dict(zip(names, row, strict=True)) for row in rows]


def _risk(prediction_s: float) -> str:
    # api.risk in the e2e config
    if -60 <= prediction_s <= 120:
        return "green"
    return "red" if prediction_s > 300 else "yellow"


def test_api_scores_every_bus_through_the_predictor(stack: Stack, replayed: Replayed) -> None:
    rows = _predictions(stack)
    buses = {b.tr_id: b for b in synthetic.BUSES}
    assert {int(r["tr_id"]) for r in rows} == set(buses)
    for row in rows:
        bus = buses[int(row["tr_id"])]
        t = parse_time(row["t"])
        assert row["sample_id"] == f"{bus.tr_id}_{int(t.timestamp())}"
        # target: the first planned stop in (T + 10 min, T + 15 min]
        k = int(row["target_stop_id"]) - bus.tr_id * 1000
        target = parse_time(row["target_time_begin"])
        assert target == synthetic.time_plan(k)
        assert t + timedelta(minutes=10) < target <= t + timedelta(minutes=15)
        assert k == 0 or synthetic.time_plan(k - 1) <= t + timedelta(minutes=10)
        assert row["risk_level"] == _risk(float(row["prediction_s"]))
    answered = [r for r in rows if r["model_version"] != "fallback"]
    # a slow first call may time out into the fallback; the rest must be the model
    assert len(answered) >= len(rows) - len(buses), [r["model_version"] for r in rows]
    model = stack.json(f"{PREDICTOR_URL}/model")["model_version"]
    assert {r["model_version"] for r in answered} <= {model, "baseline"}
    assert any(r["model_version"] == model for r in answered)


def test_predictions_are_published_on_the_stream(stack: Stack, replayed: Replayed) -> None:
    stored = {r["sample_id"] for r in _predictions(stack)}
    published = [p["sample_id"] for p in stack.compose.stream("predictions")]
    assert sorted(published) == sorted(stored)


def test_snapshot_carries_the_current_prediction(stack: Stack, replayed: Replayed) -> None:
    state = stack.state()
    views = {v["tr_id"]: v for v in state["vehicles"]}
    latest = {}
    for row in _predictions(stack):
        latest[int(row["tr_id"])] = row
    for bus in synthetic.BUSES:
        prediction = views[bus.tr_id]["prediction"]
        assert prediction is not None, views[bus.tr_id]
        assert prediction["sample_id"] == latest[bus.tr_id]["sample_id"]
        k = prediction["target_stop_id"] - bus.tr_id * 1000
        assert prediction["target_address"] == synthetic.address(bus, k)
    assert views[synthetic.IDLE_TR_ID]["prediction"] is None
    risk = state["summary"]["risk"]
    assert risk["none"] == 1
    assert sum(risk.values()) == len(views)


def test_card_lists_predictions_newest_first(stack: Stack, replayed: Replayed) -> None:
    bus = synthetic.BUSES[0]
    card = stack.json(f"{API_URL}/api/vehicles/{bus.tr_id}")
    times = [parse_time(p["t"]) for p in card["predictions"]]
    assert times and times == sorted(times, reverse=True)
    clock = parse_time(card["clock"])
    assert all(t >= clock - timedelta(seconds=1800) for t in times)


def test_predictions_are_checked_against_arrivals(stack: Stack, replayed: Replayed) -> None:
    checked = stack.compose.count("predictions", "actual_delay_s IS NOT NULL")
    assert checked > 0
    buses = {b.tr_id: b for b in synthetic.BUSES}
    rows = stack.compose.sql(
        "SELECT tr_id, actual_delay_s, abs_error_s, prediction_s FROM predictions"
        " WHERE actual_delay_s IS NOT NULL"
    )
    for tr_id, actual, error, prediction in rows:
        assert float(actual) == buses[int(tr_id)].delay_s
        assert float(error) == pytest.approx(abs(float(prediction) - float(actual)))
    state = stack.state()
    assert state["checked_predictions"] == checked
    assert state["live_mae_s"] is not None


def test_metrics(stack: Stack, replayed: Replayed) -> None:
    metrics = stack.json(f"{API_URL}/metrics")
    for key in (
        "predict_latency_ms",
        "scoring_tick_ms",
        "stream_lag_s",
        "vehicles_active",
        "predictions_per_min",
        "degraded_share",
        "alerts",
        "live_mae_s",
    ):
        assert key in metrics
    assert metrics["checked_predictions"] == stack.state()["checked_predictions"]


def test_alerts_endpoint(stack: Stack, replayed: Replayed) -> None:
    alerts = stack.json(f"{API_URL}/api/alerts?status=open")
    assert isinstance(alerts, list)
    assert len(alerts) == len(stack.state()["alerts"])


def test_stops_endpoint(stack: Stack, replayed: Replayed) -> None:
    stops = stack.json(f"{API_URL}/api/stops")
    assert len(stops) == len(synthetic.BUSES) * synthetic.PLANNED_STOPS


# Target behaviour from docs/specs/backend-design.md that the api does not
# implement yet (the endpoint table in §7).
@pytest.mark.xfail(reason="backend-design.md §7: GET /api/routes is not implemented", strict=False)
def test_routes_endpoint(stack: Stack, replayed: Replayed) -> None:
    # empty until matcher fills route_shapes, but it must answer
    assert isinstance(stack.json(f"{API_URL}/api/routes"), list)
