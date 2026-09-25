import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress

from fastapi import FastAPI, WebSocket

from app.ws import Hub, relay
from common.bus import DASHBOARD_CHANNEL
from common.config import ServiceSettings


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = ServiceSettings()
    app.state.hub = Hub()
    task = asyncio.create_task(relay(settings.redis_url, DASHBOARD_CHANNEL, app.state.hub))
    try:
        yield
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task


app = FastAPI(title="api", lifespan=lifespan)


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
