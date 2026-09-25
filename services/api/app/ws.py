"""Fan-out from the Redis bus to dashboard WebSockets.

One Redis subscription per api process, held by a background task and
shared by every client: a subscription per client would multiply Redis
connections by the number of open dashboards.
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from contextlib import aclosing, suppress

from fastapi import WebSocket
from redis.asyncio import Redis

from common.bus import subscribe

log = logging.getLogger(__name__)


class Hub:
    """The WebSocket clients currently connected to this process."""

    def __init__(self, send_timeout: float = 1.0) -> None:
        self._clients: set[WebSocket] = set()
        self._send_timeout = send_timeout

    def __len__(self) -> int:
        return len(self._clients)

    def add(self, websocket: WebSocket) -> None:
        self._clients.add(websocket)

    def discard(self, websocket: WebSocket) -> None:
        self._clients.discard(websocket)

    async def broadcast(self, text: str) -> None:
        """Send to every client in parallel.

        A client that is slow or gone is dropped and closed; the others
        are not held up by it.
        """
        await asyncio.gather(*(self._send(ws, text) for ws in list(self._clients)))

    async def _send(self, websocket: WebSocket, text: str) -> None:
        try:
            async with asyncio.timeout(self._send_timeout):
                await websocket.send_text(text)
        except Exception as exc:
            log.info("dropping websocket client: %r", exc)
            self.discard(websocket)
            with suppress(Exception):
                async with asyncio.timeout(self._send_timeout):
                    await websocket.close(code=1011)


async def relay(
    redis_url: str,
    channel: str,
    hub: Hub,
    *,
    initial_delay: float = 0.5,
    max_delay: float = 10.0,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> None:
    """Forward every bus message to the hub until cancelled.

    Losing Redis is not fatal: the task reconnects with a doubling pause,
    reset once a message gets through, and WebSocket clients stay
    connected meanwhile. Any failure short of cancellation is handled the
    same way and logged with its type, so a misconfiguration (say, a
    malformed ``redis_url``) shows up in the logs instead of silently
    ending fan-out for the rest of the process's life.
    """
    delay = initial_delay
    while True:
        client: Redis | None = None
        try:
            client = Redis.from_url(redis_url)
            async with aclosing(subscribe(client, channel)) as messages:
                async for message in messages:
                    delay = initial_delay
                    await hub.broadcast(message.model_dump_json())
        except Exception as exc:
            log.warning(
                "bus subscription lost (%s: %s), retrying in %.1fs",
                type(exc).__name__,
                exc,
                delay,
            )
        finally:
            if client is not None:
                await client.aclose()
        await sleep(delay)
        delay = min(delay * 2, max_delay)
