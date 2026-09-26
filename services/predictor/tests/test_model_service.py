import math
from datetime import timedelta

import pytest
from fastapi.testclient import TestClient

from app.config import PredictorConfig, PredictorSettings
from app.examples import START
from app.examples import bus_request as request
from app.main import create_app
from app.predictor import REASON_CODES, reason_codes, to_query, to_response
from busdelay.explain import REASONS
from busdelay.inference import Answer
from common.config import REPO_ROOT, load_section
from contracts import PredictRequest, PredictResponse, ReasonCode

SYSTEM_YAML = REPO_ROOT / "config" / "system.yaml"
MODEL_DIR = REPO_ROOT / "models" / "current"


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


def silent(r: PredictRequest, minutes: float) -> PredictRequest:
    """The same request with the last ``minutes`` of telemetry lost."""
    cut = r.T - timedelta(minutes=minutes)
    return r.model_copy(update={"telemetry": [p for p in r.telemetry if p.event_time <= cut]})


def blind(r: PredictRequest) -> PredictRequest:
    """The same request with every fix reported without a position."""
    lost = {"lat": None, "lon": None, "location_valid": False}
    return r.model_copy(update={"telemetry": [p.model_copy(update=lost) for p in r.telemetry]})


@pytest.mark.parametrize(
    "lose",
    [
        lambda r: silent(r, 10),
        blind,
        lambda r: r.model_copy(update={"telemetry": []}),
    ],
    ids=["silent for 10 min", "no position", "no telemetry"],
)
def test_lost_telemetry_is_forecast_by_the_model_and_flagged(client, lose):
    fresh = request(1, 90.0, cur_dev_s=80.0)
    answers = post(client, [fresh, lose(request(2, 90.0, cur_dev_s=80.0))])

    assert all(a.model_version.startswith("current@") for a in answers)
    assert all(math.isfinite(a.prediction_s) and a.p_late is not None for a in answers)
    assert ReasonCode.stale_telemetry not in answers[0].reasons
    assert ReasonCode.stale_telemetry in answers[1].reasons


def test_stale_telemetry_is_added_by_the_age_of_the_last_position():
    def reasons(age: float) -> list[ReasonCode]:
        answer = Answer("s", 60.0, 0.3, ["carried_delay"], 50.0, position_age_s=age)
        return to_response(answer, "v", stale_after_s=120.0).reasons

    assert reasons(30.0) == [ReasonCode.accumulated_delay]
    assert reasons(120.0) == [ReasonCode.accumulated_delay]
    assert reasons(121.0) == [ReasonCode.accumulated_delay, ReasonCode.stale_telemetry]
    assert reasons(math.nan) == [ReasonCode.accumulated_delay, ReasonCode.stale_telemetry]
    alone = Answer("s", 0.0, 0.1, [], 0.0, position_age_s=math.nan)
    assert to_response(alone, "v", 120.0).reasons == [ReasonCode.stale_telemetry]


def test_the_swagger_example_gets_the_model(client):
    schema = client.get("/openapi.json").json()
    body = schema["paths"]["/predict"]["post"]["requestBody"]["content"]["application/json"]
    response = client.post("/predict", json=body["examples"]["late_bus"]["value"])

    assert response.status_code == 200
    answer = PredictResponse.model_validate(response.json()[0])
    assert answer.model_version.startswith("current@")
    assert answer.prediction_s > 120 and ReasonCode.accumulated_delay in answer.reasons


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
    assert config.stale_after_sec == 120
