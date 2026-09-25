from datetime import timedelta

from factories import T0
from fastapi.testclient import TestClient

from app.main import app
from app.schemas import WS_MESSAGE, ClockMessage, PredictionMessage, VehiclesMessage


def test_ws_message_is_told_apart_by_type() -> None:
    clock = WS_MESSAGE.validate_json(
        '{"type": "clock", "data": {"seq": 3, "clock": "2026-01-06T08:00:00Z"}}'
    )
    vehicles = WS_MESSAGE.validate_json(
        '{"type": "vehicles", "data": {"seq": 4, "clock": "2026-01-06T08:00:00Z",'
        ' "vehicles": [], "removed": [7]}}'
    )

    assert isinstance(clock, ClockMessage)
    assert (clock.data.seq, clock.data.clock) == (3, T0)
    assert isinstance(vehicles, VehiclesMessage)
    assert vehicles.data.removed == [7]


def test_ws_message_serializes_times_as_utc() -> None:
    message = ClockMessage(type="clock", data={"seq": 1, "clock": T0 + timedelta(seconds=1)})

    assert message.model_dump_json() == (
        '{"type":"clock","data":{"seq":1,"clock":"2026-01-06T08:00:01Z"}}'
    )


def test_openapi_exposes_ws_message_with_its_parts() -> None:
    schema = TestClient(app).get("/openapi.json").json()
    components = schema["components"]["schemas"]
    ws_message = components["WsMessage"]

    assert ws_message["discriminator"]["propertyName"] == "type"
    assert set(ws_message["discriminator"]["mapping"]) == {
        "clock",
        "vehicles",
        "prediction",
        "alert",
    }
    for ref in ws_message["discriminator"]["mapping"].values():
        assert ref.removeprefix("#/components/schemas/") in components
    for name in ("VehicleView", "PredictionView", "AlertView", "ClockData", "VehiclesData"):
        assert name in components


def test_prediction_message_round_trips() -> None:
    raw = (
        '{"type":"prediction","data":{"seq":5,"tr_id":7,"prediction":{'
        '"sample_id":"7_1","t":"2026-01-06T08:00:00Z","target_stop_id":2,'
        '"target_address":null,"target_time_begin":"2026-01-06T08:12:00Z",'
        '"prediction_s":400.0,"p_late":null,"reasons":["accumulated_delay"],'
        '"risk_level":"red","degraded":false,"degraded_reason":null,"model_version":"m1"}}}'
    )

    message = WS_MESSAGE.validate_json(raw)

    assert isinstance(message, PredictionMessage)
    assert message.model_dump_json() == raw
