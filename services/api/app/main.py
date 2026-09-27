import asyncio
import logging
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from datetime import datetime
from typing import Annotated, Any

import psycopg
from fastapi import FastAPI, HTTPException, Query, Request, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from psycopg_pool import ConnectionPool
from redis.asyncio import Redis

from app.config import ApiConfig
from app.dashboard import Dashboard, StateUnavailable, VehicleNotFound
from app.loop import run_prediction_loop
from app.metrics import LoopMetrics
from app.models import AlertRow, AlertStatus, PredictionRow
from app.predictor_client import PredictorClient
from app.schemas import (
    AlertStats,
    AlertView,
    Metrics,
    StateSnapshot,
    VehicleCard,
    ws_message_schemas,
)
from app.store import AlertCounts, count_alerts, load_alerts, load_vehicle_predictions
from app.views import alert_view
from app.ws import Hub, relay
from common.bus import DASHBOARD_CHANNEL
from common.config import ServiceSettings, load_section
from common.db import make_pool

# How long an endpoint waits for a database connection before answering 503.
DB_TIMEOUT_SEC = 2.0


class _AppLogHandler(logging.Handler):
    """Writes formatted records to ``sys.stderr`` as it is at emit time.

    A plain ``logging.StreamHandler()`` freezes the stream object at
    construction; pytest swaps ``sys.stderr`` for each test's capture, so a
    handler built earlier would then write to an already-closed stream.
    Reading ``sys.stderr`` inside ``emit`` instead keeps it always current.
    """

    def emit(self, record: logging.LogRecord) -> None:
        try:
            sys.stderr.write(self.format(record) + "\n")
        except Exception:
            self.handleError(record)


def configure_logging() -> None:
    """Give the ``app`` logger a level and a formatted handler.

    uvicorn configures only its own loggers (``uvicorn``, ``uvicorn.error``,
    ``uvicorn.access``), which do not propagate to the root, so ``app.*``
    records would otherwise reach Python's last-resort handler and print as
    a bare message with no level or logger name. The root logger is left
    alone: raising its level to INFO would also print httpx's INFO line for
    every predictor request. Idempotent: a second call does not add a
    second handler.
    """
    logger = logging.getLogger("app")
    if not any(isinstance(handler, _AppLogHandler) for handler in logger.handlers):
        handler = _AppLogHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    configure_logging()
    settings = ServiceSettings()
    config = _config
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
    app.state.metrics = LoopMetrics()
    predictor = PredictorClient(settings.predictor_url, config.predict_timeout_ms / 1000)
    tasks = [
        asyncio.create_task(relay(settings.redis_url, DASHBOARD_CHANNEL, app.state.hub)),
        asyncio.create_task(
            run_prediction_loop(
                pool,
                settings.redis_url,
                predictor,
                config,
                dashboard=app.state.dashboard,
                metrics=app.state.metrics,
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


class _ConfigLocation(ServiceSettings):
    """Only where the config file is: connection URLs are checked at startup."""

    database_url: str | None = None
    redis_url: str | None = None


# Read at import, not in the lifespan: middleware cannot be added once the
# app has been called, and the lifespan is itself such a call.
_config = load_section(_ConfigLocation().config_path, "api", ApiConfig)

app = FastAPI(title="api", lifespan=lifespan)
# The dashboard is served from another origin (backend-design.md §7).
app.add_middleware(
    CORSMiddleware,
    allow_origins=_config.cors_origins,
    allow_methods=["GET"],
    allow_credentials=False,
)
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
    with pool.connection(timeout=DB_TIMEOUT_SEC) as conn:
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


def _alerts(pool: ConnectionPool, status: AlertStatus | None, limit: int) -> list[AlertRow]:
    with pool.connection(timeout=DB_TIMEOUT_SEC) as conn:
        return load_alerts(conn, status, limit)


@app.get("/api/alerts", responses={503: {"description": "Postgres did not answer"}})
async def get_alerts(
    request: Request,
    status: AlertStatus | None = None,
    limit: Annotated[int, Query(ge=1, le=1000)] = 200,
) -> list[AlertView]:
    """Alerts with the given status, all if none is given, newest first."""
    try:
        rows = await asyncio.to_thread(_alerts, request.app.state.pool, status, limit)
    except psycopg.Error as exc:
        raise HTTPException(503, "alerts are not available") from exc
    return [alert_view(row) for row in rows]


def _alert_counts(pool: ConnectionPool) -> AlertCounts:
    with pool.connection(timeout=DB_TIMEOUT_SEC) as conn:
        return count_alerts(conn)


@app.get("/metrics", responses={503: {"description": "Postgres did not answer"}})
async def get_metrics(request: Request) -> Metrics:
    """How the prediction loop is doing: timings, rates, alerts and the live MAE.

    Timings, rates and the degraded share cover the last five minutes of
    real time; alerts and the live MAE cover the whole run.
    """
    dashboard: Dashboard = request.app.state.dashboard
    metrics: LoopMetrics = request.app.state.metrics
    try:
        alerts = await asyncio.to_thread(_alert_counts, request.app.state.pool)
    except psycopg.Error as exc:
        raise HTTPException(503, "metrics are not available") from exc
    snapshot = dashboard.snapshot()
    return Metrics(
        predict_latency_ms=metrics.predict_latency_ms(),
        predict_failures=metrics.predict_failures(),
        scoring_tick_ms=metrics.scoring_tick_ms(),
        stream_lag_s=metrics.stream_lag_s,
        vehicles_active=snapshot.summary.freshness.active,
        predictions_per_min=metrics.predictions_per_min(),
        degraded_share=metrics.degraded_share(),
        alerts=AlertStats(**alerts._asdict()),
        live_mae_s=snapshot.live_mae_s,
        checked_predictions=snapshot.checked_predictions,
    )


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
