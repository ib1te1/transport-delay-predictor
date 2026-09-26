import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from datetime import datetime
from typing import Any

import psycopg
from fastapi import FastAPI, HTTPException, Request, WebSocket
from psycopg_pool import ConnectionPool
from redis.asyncio import Redis

from app.config import ApiConfig
from app.dashboard import Dashboard, StateUnavailable, VehicleNotFound
from app.loop import run_prediction_loop
from app.models import PredictionRow
from app.predictor_client import PredictorClient
from app.schemas import StateSnapshot, VehicleCard, ws_message_schemas
from app.store import load_vehicle_predictions
from app.ws import Hub, relay
from common.bus import DASHBOARD_CHANNEL
from common.config import ServiceSettings, load_section
from common.db import make_pool

# How long a card waits for a database connection before answering 503.
CARD_DB_TIMEOUT_SEC = 2.0


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = ServiceSettings()
    config = load_section(settings.config_path, "api", ApiConfig)
    app.state.hub = Hub()
    # Opened without waiting: the loop retries until Postgres answers, and
    # the API must come up even while it does not.
    pool = make_pool(settings.database_url)
    pool.open(wait=False)
    app.state.pool = pool
    # A client of its own: the loop's is replaced on every restart, while
    # the dashboard's numbering and published view last as long as the process.
    bus = Redis.from_url(settings.redis_url)
    app.state.dashboard = Dashboard(bus, config)
    predictor = PredictorClient(settings.predictor_url, config.predict_timeout_ms / 1000)
    tasks = [
        asyncio.create_task(relay(settings.redis_url, DASHBOARD_CHANNEL, app.state.hub)),
        asyncio.create_task(
            run_prediction_loop(
                pool, settings.redis_url, predictor, config, dashboard=app.state.dashboard
            )
        ),
    ]
    try:
        yield
    finally:
        for task in tasks:
            task.cancel()
        for task in tasks:
            with suppress(asyncio.CancelledError):
                await task
        await predictor.aclose()
        await bus.aclose()
        await asyncio.to_thread(pool.close)


app = FastAPI(title="api", lifespan=lifespan)
_fastapi_openapi = app.openapi


def openapi_with_ws_messages() -> dict[str, Any]:
    """FastAPI's schema plus ``WsMessage`` and its parts, built once and cached.

    No endpoint returns ``WsMessage``, so FastAPI would leave it out; the
    frontend generates the WebSocket message types from here all the same.
    """
    if app.openapi_schema is None:
        schema = _fastapi_openapi()
        components = schema.setdefault("components", {}).setdefault("schemas", {})
        for name, definition in ws_message_schemas("#/components/schemas/{model}").items():
            components.setdefault(name, definition)
        app.openapi_schema = schema
    return app.openapi_schema


app.openapi = openapi_with_ws_messages


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "api"}


@app.get("/api/state")
async def get_state(request: Request) -> StateSnapshot:
    """The dashboard state as last published over ``/ws``, with that message's ``seq``.

    Answers even before the prediction loop is up: then ``seq`` is 0,
    ``clock`` is null and the lists are empty.
    """
    dashboard: Dashboard = request.app.state.dashboard
    return dashboard.snapshot()


def _vehicle_predictions(pool: ConnectionPool, tr_id: int, since: datetime) -> list[PredictionRow]:
    with pool.connection(timeout=CARD_DB_TIMEOUT_SEC) as conn:
        return load_vehicle_predictions(conn, tr_id, since)


@app.get(
    "/api/vehicles/{tr_id}",
    responses={
        404: {"description": "The vehicle is neither in the plan nor in the telemetry window"},
        503: {"description": "The prediction loop is not up yet, or Postgres did not answer"},
    },
)
async def get_vehicle(tr_id: int, request: Request) -> VehicleCard:
    """One vehicle in detail: its view, recent predictions, track and planned stops nearby."""
    dashboard: Dashboard = request.app.state.dashboard
    pool: ConnectionPool = request.app.state.pool

    async def load(vehicle: int, since: datetime) -> list[PredictionRow]:
        return await asyncio.to_thread(_vehicle_predictions, pool, vehicle, since)

    try:
        return await dashboard.card(tr_id, load)
    except VehicleNotFound as exc:
        raise HTTPException(404, f"vehicle {tr_id} is not planned and not seen") from exc
    except (StateUnavailable, psycopg.Error) as exc:
        raise HTTPException(503, "vehicle state is not available yet") from exc


@app.websocket("/ws")
async def ws(websocket: WebSocket) -> None:
    """Push-only: bus messages go out, whatever the client sends is ignored.

    State is not sent here; after (re)connecting the dashboard fetches it
    over REST. Keepalive ping/pong is handled by uvicorn.
    """
    hub: Hub = websocket.app.state.hub
    await websocket.accept()
    hub.add(websocket)
    try:
        while (await websocket.receive())["type"] != "websocket.disconnect":
            pass
    finally:
        hub.discard(websocket)
