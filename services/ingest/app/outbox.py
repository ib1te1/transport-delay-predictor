"""Deliver persisted telemetry to Redis Streams with at-least-once semantics."""

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable

from psycopg_pool import ConnectionPool
from redis.asyncio import Redis

from app.store import is_published, mark_published, pending_backlog, pending_telemetry
from common.bus import TELEMETRY_STREAM, append_async

log = logging.getLogger(__name__)

type Sleep = Callable[[float], Awaitable[None]]


def _pending(pool: ConnectionPool):
    with pool.connection() as conn:
        return pending_telemetry(conn)


def _mark(pool: ConnectionPool, row_id: int, stream_id: str) -> None:
    with pool.connection() as conn:
        mark_published(conn, row_id, stream_id)


def _published(pool: ConnectionPool, row_id: int) -> bool:
    with pool.connection() as conn:
        return is_published(conn, row_id)


def _backlog(pool: ConnectionPool) -> tuple[int, float | None]:
    with pool.connection() as conn:
        return pending_backlog(conn)


class Outbox:
    """One publisher per process; callers can wake it after inserting rows."""

    def __init__(
        self,
        pool: ConnectionPool,
        redis: Redis,
        *,
        stream: str = TELEMETRY_STREAM,
        retry_sec: float = 1.0,
    ) -> None:
        self.pool = pool
        self.redis = redis
        self.stream = stream
        self.retry_sec = retry_sec
        self._wake = asyncio.Event()
        self._lock = asyncio.Lock()
        self._last_backlog_log = 0.0

    def notify(self) -> None:
        self._wake.set()

    async def flush(self) -> int:
        """Publish everything currently pending, oldest database row first."""
        delivered = 0
        async with self._lock:
            while True:
                pending = await asyncio.to_thread(_pending, self.pool)
                if not pending:
                    return delivered
                now = time.monotonic()
                if now - self._last_backlog_log >= 60:
                    count, age = await asyncio.to_thread(_backlog, self.pool)
                    log.info("telemetry outbox pending=%d oldest_age_sec=%s", count, age)
                    self._last_backlog_log = now
                for row in pending:
                    stream_id = await append_async(self.redis, self.stream, row.record)
                    await asyncio.to_thread(_mark, self.pool, row.id, stream_id)
                    delivered += 1

    async def wait_for(self, row_id: int, *, sleep: Sleep = asyncio.sleep) -> None:
        """Hold CSV playback at this row while Redis is unavailable."""
        while True:
            try:
                await self.flush()
                if await asyncio.to_thread(_published, self.pool, row_id):
                    return
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.warning("telemetry delivery failed, retrying: %s", exc)
            await sleep(self.retry_sec)

    async def drain(self, *, sleep: Sleep = asyncio.sleep) -> None:
        """Retry startup recovery until all rows saved before a crash are published."""
        while True:
            try:
                await self.flush()
                return
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.warning("pending telemetry delivery failed, retrying: %s", exc)
                await sleep(self.retry_sec)

    async def run(self) -> None:
        """Keep the NDTP outbox draining without blocking the TCP listener."""
        while True:
            try:
                delivered = await self.flush()
                if delivered:
                    continue
                self._wake.clear()
                try:
                    await asyncio.wait_for(self._wake.wait(), timeout=self.retry_sec)
                except TimeoutError:
                    pass
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.warning("telemetry outbox unavailable, retrying: %s", exc)
                await asyncio.sleep(self.retry_sec)
