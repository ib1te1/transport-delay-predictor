from fastapi.testclient import TestClient
from test_predict import request

from app.main import create_app
from app.metrics import PredictTimings


def test_metrics_before_any_call() -> None:
    body = TestClient(create_app()).get("/metrics").json()

    assert body["model_version"] == "baseline"
    assert body["predict"] == {
        "calls": 0,
        "failed": 0,
        "window": 0,
        "latency_ms": None,
        "batch_size": None,
        "body_kb": None,
    }


def test_metrics_time_each_predict_call() -> None:
    client = TestClient(create_app())
    assert client.post("/predict", json=[request(1, 60.0), request(2, None)]).status_code == 200
    assert client.post("/predict", json=[request(3, 0.0)]).status_code == 200
    assert client.post("/predict", json=[{"sample_id": "broken"}]).status_code == 422
    client.get("/health")

    predict = client.get("/metrics").json()["predict"]

    assert (predict["calls"], predict["failed"], predict["window"]) == (3, 1, 2)
    assert predict["batch_size"]["max"] == 2
    assert 0 < predict["latency_ms"]["p50"] <= predict["latency_ms"]["max"]
    assert predict["body_kb"]["max"] > 0


def test_timings_keep_only_the_window() -> None:
    timings = PredictTimings(window=3)
    for ms in (900.0, 10.0, 20.0, 30.0):
        timings.add(ms, ok=True, batch=1, body_bytes=2048)

    summary = timings.summary()

    assert (summary.calls, summary.window) == (4, 3)
    assert summary.latency_ms.max == 30.0
    assert summary.latency_ms.p50 == 20.0
    assert summary.body_kb.max == 2.0
