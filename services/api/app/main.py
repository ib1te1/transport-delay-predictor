import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from typing import Any

from fastapi import FastAPI, WebSocket
from fastapi.openapi.utils import get_openapi

from app.config import ApiConfig
from app.loop import run_prediction_loop
from app.predictor_client import PredictorClient
from app.schemas import ws_message_schemas
from app.ws import Hub, relay
from common.bus import DASHBOARD_CHANNEL
from common.config import ServiceSettings, load_section
from common.db import make_pool


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = ServiceSettings()
    config = load_section(settings.config_path, "api", ApiConfig)
    app.state.hub = Hub()
    # Opened without waiting: the loop retries until Postgres answers, and
    # the API must come up even while it does not.
    pool = make_pool(settings.database_url)
    pool.open(wait=False)
    predictor = PredictorClient(settings.predictor_url, config.predict_timeout_ms / 1000)
    tasks = [
        asyncio.create_task(relay(settings.redis_url, DASHBOARD_CHANNEL, app.state.hub)),
        asyncio.create_task(run_prediction_loop(pool, settings.redis_url, predictor, config)),
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
        await asyncio.to_thread(pool.close)


app = FastAPI(title="api", lifespan=lifespan)


def openapi_with_ws_messages() -> dict[str, Any]:
    """FastAPI's schema plus ``WsMessage`` and its parts, built once and cached.

    No endpoint returns ``WsMessage``, so FastAPI would leave it out; the
    frontend generates the WebSocket message types from here all the same.
    """
    if app.openapi_schema is None:
        schema = get_openapi(title=app.title, version=app.version, routes=app.routes)
        components = schema.setdefault("components", {}).setdefault("schemas", {})
        for name, definition in ws_message_schemas("#/components/schemas/{model}").items():
            components.setdefault(name, definition)
        app.openapi_schema = schema
    return app.openapi_schema


app.openapi = openapi_with_ws_messages


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "api"}


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
