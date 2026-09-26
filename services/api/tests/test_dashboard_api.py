import asyncio
import random
from datetime import timedelta

import psycopg
import pytest
from factories import at, plan_stop, prediction_row, telemetry_record
from fastapi.testclient import TestClient

import app.main as main_module
from app.config import ApiConfig
from app.dashboard import Dashboard
from app.live import LiveState
from app.main import app
from app.schemas import ws_message_schemas
from app.state import FleetState
from app.store import save_predictions
from common.db import connect

CONFIG = ApiConfig()
WINDOW = timedelta(seconds=CONFIG.request.telemetry_window_sec)


class Bus:
    """Accepts every publish and keeps nothing."""

    async def publish(self, channel: str, message: str) -> int:
        return 0


def live_state(tr_id: int) -> LiveState:
    """Vehicle ``tr_id`` seen at at(0) and at(60), with one planned stop 20 at at(720)."""
    fleet = FleetState(WINDOW)
    fleet.add_telemetry(telemetry_record(tr_id, at(0)))
    fleet.add_telemetry(telemetry_record(tr_id, at(60)))
    return LiveState(fleet, {tr_id: [plan_stop(tr_id, 20, at(720), address="Lenina 1")]})


@pytest.fixture
def published(monkeypatch: pytest.MonkeyPatch) -> Dashboard:
    """``app`` without its lifespan, serving a dashboard that has published vehicle 7."""
    dashboard = Dashboard(Bus(), CONFIG)
    dashboard.attach(live_state(7))
    asyncio.run(dashboard.refresh())
    monkeypatch.setattr(app.state, "dashboard", dashboard, raising=False)
    monkeypatch.setattr(app.state, "pool", None, raising=False)
    return dashboard


def test_state_returns_the_last_published_snapshot(published: Dashboard) -> None:
    response = TestClient(app).get("/api/state")

    assert response.status_code == 200
    body = response.json()
    assert (body["seq"], body["clock"]) == (2, "2026-01-06T08:01:00Z")
    assert [v["tr_id"] for v in body["vehicles"]] == [7]
    assert body["vehicles"][0]["freshness"] == "active"
    assert body["summary"] == {
        "risk": {"green": 0, "yellow": 0, "red": 0, "none": 1},
        "freshness": {"active": 1, "stale": 0, "offline": 0},
    }
    assert (body["alerts"], body["live_mae_s"], body["checked_predictions"]) == ([], None, 0)


def test_card_of_an_unknown_vehicle_is_404(published: Dashboard) -> None:
    response = TestClient(app).get("/api/vehicles/99")

    assert response.status_code == 404
    assert response.json() == {"detail": "vehicle 99 is not planned and not seen"}


def test_card_is_503_when_postgres_does_not_answer(
    published: Dashboard, monkeypatch: pytest.MonkeyPatch
) -> None:
    def down(pool, tr_id, since):
        raise psycopg.OperationalError("connection refused")

    monkeypatch.setattr(main_module, "_vehicle_predictions", down)

    response = TestClient(app).get("/api/vehicles/7")

    assert response.status_code == 503
    assert response.json() == {"detail": "vehicle state is not available yet"}


def test_card_returns_the_vehicle_its_predictions_track_and_stops(
    published: Dashboard, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []

    def stored(pool, tr_id, since):
        calls.append((tr_id, since))
        return [prediction_row(sample_id="7_60", tr_id=7, t=at(60), target_stop_id=20)]

    monkeypatch.setattr(main_module, "_vehicle_predictions", stored)

    response = TestClient(app).get("/api/vehicles/7")

    assert response.status_code == 200
    body = response.json()
    assert (body["seq"], body["clock"]) == (2, "2026-01-06T08:01:00Z")
    assert calls == [(7, at(60) - timedelta(seconds=CONFIG.card_track_sec))]
    assert body["vehicle"]["tr_id"] == 7
    assert [p["sample_id"] for p in body["predictions"]] == ["7_60"]
    assert body["predictions"][0]["target_address"] == "Lenina 1"
    assert [p["event_time"] for p in body["track"]] == [
        "2026-01-06T08:00:00Z",
        "2026-01-06T08:01:00Z",
    ]
    assert body["stops"] == [
        {
            "stop_id": 20,
            "address": "Lenina 1",
            "lat": 55.75,
            "lon": 37.62,
            "time_plan": "2026-01-06T08:12:00Z",
            "time_fact": None,
            "delay_s": None,
        }
    ]


def test_openapi_keeps_app_level_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main_module.app, "description", "probe")
    monkeypatch.setattr(main_module.app, "openapi_schema", None)

    schema = TestClient(app).get("/openapi.json").json()

    assert schema["info"]["description"] == "probe"
    assert "WsMessage" in schema["components"]["schemas"]


def test_openapi_describes_the_dashboard_endpoints() -> None:
    schema = TestClient(app).get("/openapi.json").json()
    state = schema["paths"]["/api/state"]["get"]["responses"]["200"]
    card = schema["paths"]["/api/vehicles/{tr_id}"]["get"]["responses"]
    components = schema["components"]["schemas"]

    assert state["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/StateSnapshot"
    }
    assert {"200", "404", "503"} <= set(card)
    # REST responses and WebSocket messages share one definition of each view.
    shared = ws_message_schemas("#/components/schemas/{model}")
    for name in ("VehicleView", "PredictionView", "AlertView", "ReasonCode"):
        assert components[name] == shared[name]


# database_url and redis_url are not used directly: TestClient(app) runs the
# lifespan, which builds ServiceSettings, where both are required.
@pytest.mark.usefixtures("database_url", "redis_url", "no_prediction_loop")
@pytest.mark.timeout(30)
def test_before_the_loop_is_up_state_is_empty_and_the_card_is_503() -> None:
    with TestClient(app) as client:
        state = client.get("/api/state")
        card = client.get("/api/vehicles/7")

    assert state.status_code == 200
    assert state.json() == {
        "seq": 0,
        "clock": None,
        "vehicles": [],
        "summary": {
            "risk": {"green": 0, "yellow": 0, "red": 0, "none": 0},
            "freshness": {"active": 0, "stale": 0, "offline": 0},
        },
        "alerts": [],
        "live_mae_s": None,
        "checked_predictions": 0,
    }
    assert card.status_code == 503


@pytest.mark.usefixtures("redis_url", "no_prediction_loop")
@pytest.mark.timeout(30)
def test_card_reads_the_vehicle_predictions_from_postgres(database_url: str) -> None:
    # Far above anything in the real seed data; the row is deleted below.
    tr_id = 9_000_000_000 + random.randrange(1_000_000)
    stored = prediction_row(sample_id=f"{tr_id}_60", tr_id=tr_id, t=at(60), target_stop_id=20)
    with connect(database_url) as conn:
        save_predictions(conn, [stored])
        conn.commit()
    try:
        with TestClient(app) as client:
            dashboard: Dashboard = client.app.state.dashboard
            dashboard.attach(live_state(tr_id))
            # The dashboard belongs to the app's event loop, run by TestClient's portal.
            client.portal.call(dashboard.refresh)
            response = client.get(f"/api/vehicles/{tr_id}")
    finally:
        with connect(database_url) as conn:
            conn.execute("DELETE FROM predictions WHERE tr_id = %s", (tr_id,))
            conn.commit()

    assert response.status_code == 200
    body = response.json()
    assert body["vehicle"]["tr_id"] == tr_id
    assert [p["sample_id"] for p in body["predictions"]] == [stored.sample_id]
