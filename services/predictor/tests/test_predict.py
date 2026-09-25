from datetime import UTC, datetime

from fastapi.testclient import TestClient

from app.main import app
from contracts import PredictRequest, PredictResponse, make_sample_id

T = datetime(2026, 1, 6, 3, 35, tzinfo=UTC)


def request(tr_id: int, cur_dev_s: float | None) -> dict:
    return PredictRequest(
        sample_id=make_sample_id(tr_id, T),
        tr_id=tr_id,
        T=T,
        target_stop_id=1,
        target_time_begin=datetime(2026, 1, 6, 3, 50, tzinfo=UTC),
        cur_dev_s=cur_dev_s,
        telemetry=[],
        schedule=[],
    ).model_dump(mode="json")


def test_predict_returns_baseline_in_request_order() -> None:
    response = TestClient(app).post("/predict", json=[request(1, 274.0), request(2, None)])

    assert response.status_code == 200
    body = [PredictResponse.model_validate(item) for item in response.json()]
    assert [(r.sample_id, r.prediction_s) for r in body] == [
        (make_sample_id(1, T), 274.0),
        (make_sample_id(2, T), 0.0),
    ]
    assert {r.model_version for r in body} == {"baseline"}


def test_predict_rejects_a_naive_timestamp() -> None:
    bad = request(1, 0.0) | {"T": "2026-01-06T03:35:00"}
    assert TestClient(app).post("/predict", json=[bad]).status_code == 422
