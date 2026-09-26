import math
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from app.config import PredictorConfig, PredictorSettings
from app.main import create_app
from app.predictor import REASON_CODES, reason_codes, to_query
from busdelay.explain import REASONS
from common.config import REPO_ROOT, load_section
from contracts import (
    PredictRequest,
    PredictResponse,
    ReasonCode,
    ScheduledStop,
    TelemetryPoint,
    make_sample_id,
)

SYSTEM_YAML = REPO_ROOT / "config" / "system.yaml"
MODEL_DIR = REPO_ROOT / "models" / "current"
START = datetime(2026, 1, 6, 4, 0, tzinfo=UTC)


def stops(tr_id: int, n: int = 120) -> list[ScheduledStop]:
    """A street to the east, a stop every ~310 m and every minute, a layover after stop 59."""
    return [
        ScheduledStop(
            stop_id=tr_id * 1000 + k,
            time_plan=START + timedelta(minutes=k + (10 if k >= 60 else 0)),
            lat=55.75 + tr_id * 0.01,
            lon=37.60 + 0.005 * k,
            time_fact=None,
        )
        for k in range(n)
    ]


def fixes(schedule: list[ScheduledStop], late_s: float, until: datetime) -> list[TelemetryPoint]:
    """A bus that reaches every stop ``late_s`` after the plan, a fix every 10 s."""
    points = []
    for a, b in zip(schedule, schedule[1:], strict=False):
        leave, arrive = a.time_plan + timedelta(seconds=late_s + 20), b.time_plan
        arrive += timedelta(seconds=late_s)
        t = a.time_plan + timedelta(seconds=late_s)
        while t < arrive and t <= until:
            share = 0.0 if t <= leave else (t - leave) / (arrive - leave)
            points.append(
                TelemetryPoint(
                    event_time=t,
                    lat=a.lat,
                    lon=a.lon + (b.lon - a.lon) * share,
                    location_valid=True,
                    speed_kmh=0.0 if t <= leave else 30.0,
                    heading_deg=90.0,
                )
            )
            t += timedelta(seconds=10)
    return points


def request(tr_id: int, late_s: float, cur_dev_s: float | None = None) -> PredictRequest:
    T = START + timedelta(minutes=40)
    schedule = stops(tr_id)
    target = next(s for s in schedule if s.time_plan > T + timedelta(minutes=10))
    return PredictRequest(
        sample_id=make_sample_id(tr_id, T),
        tr_id=tr_id,
        T=T,
        target_stop_id=target.stop_id,
        target_time_begin=target.time_plan,
        cur_dev_s=cur_dev_s,
        telemetry=fixes(schedule, late_s, T),
        schedule=[s for s in schedule if s.time_plan <= target.time_plan],
    )


def post(client: TestClient, requests: list[PredictRequest]) -> list[PredictResponse]:
    body = [r.model_dump(mode="json") for r in requests]
    response = client.post("/predict", json=body)
    assert response.status_code == 200
    return [PredictResponse.model_validate(item) for item in response.json()]


@pytest.fixture(scope="module")
def client():
    app = create_app(PredictorSettings(model_dir=MODEL_DIR, config_path=SYSTEM_YAML))
    with TestClient(app) as client:
        yield client


def test_the_model_answers_in_request_order(client):
    answers = post(client, [request(1, 90.0), request(2, 240.0)])

    assert [a.sample_id for a in answers] == [request(1, 90).sample_id, request(2, 240).sample_id]
    for a in answers:
        assert a.model_version.startswith("current@")
        assert math.isfinite(a.prediction_s) and -1800 < a.prediction_s < 1800
        assert a.p_late is not None and 0.0 <= a.p_late <= 1.0
        assert a.reasons and all(isinstance(r, ReasonCode) for r in a.reasons)


def test_a_point_the_model_cannot_use_gets_the_baseline(client):
    broken = request(2, 60.0, cur_dev_s=75.0).model_copy(update={"target_stop_id": -1})
    answers = post(client, [request(1, 90.0), broken])

    assert answers[0].model_version.startswith("current@")
    assert answers[1].model_version == "baseline" and answers[1].prediction_s == 75.0


def test_model_endpoint_reports_the_model(client):
    info = client.get("/model").json()
    assert info["kind"] == "model" and info["model_version"].startswith("current@")
    assert info["cur_dev_source"] == "gps"
    assert {"model", "model, cur_dev from GPS"} <= set(info["cv_mae_s"])


def test_without_a_model_the_baseline_answers(tmp_path):
    app = create_app(PredictorSettings(model_dir=tmp_path, config_path=SYSTEM_YAML))
    with TestClient(app) as client:
        answers = post(client, [request(1, 90.0, cur_dev_s=33.0)])
        info = client.get("/model").json()
    assert answers[0].model_version == "baseline" and answers[0].prediction_s == 33.0
    assert info["kind"] == "baseline" and str(tmp_path) in info["note"]


def test_a_broken_model_does_not_take_the_service_down(tmp_path):
    (tmp_path / "regressor.cbm").write_bytes(b"not a model")
    (tmp_path / "meta.json").write_text("{}", encoding="utf-8")
    app = create_app(PredictorSettings(model_dir=tmp_path, config_path=SYSTEM_YAML))
    with TestClient(app) as client:
        answers = post(client, [request(1, 90.0)])
        info = client.get("/model").json()
    assert answers[0].model_version == "baseline"
    assert "does not load" in info["note"]


def test_every_model_reason_has_a_contract_code():
    assert set(REASON_CODES) == set(REASONS)


def test_reason_codes_keep_order_and_drop_unknown_when_something_is_known():
    assert reason_codes(["carried_delay", "layover", "long_dwell"]) == [
        ReasonCode.accumulated_delay,
        ReasonCode.long_dwell,
    ]
    assert reason_codes(["layover", "timetable"]) == [ReasonCode.unknown]
    assert reason_codes([]) == [ReasonCode.unknown]


def test_request_becomes_a_query_without_facts():
    r = request(1, 90.0, cur_dev_s=None)
    r.telemetry[0] = r.telemetry[0].model_copy(update={"lat": None, "location_valid": False})
    r.schedule[0] = r.schedule[0].model_copy(update={"time_fact": START})

    query = to_query(r)

    assert query.T == r.T.timestamp() and math.isnan(query.cur_dev_s)
    assert list(query.plan.columns) == ["stop_id", "t_plan", "lat", "lon"]
    assert query.plan["t_plan"].iloc[0] == START.timestamp()
    assert math.isnan(query.fixes["lat"].iloc[0]) and not query.fixes["location_valid"].iloc[0]
    assert len(query.fixes) == len(r.telemetry)


def test_repository_config_has_a_valid_predictor_section():
    config = load_section(SYSTEM_YAML, "predictor", PredictorConfig)
    assert config.cur_dev_source == "gps"
