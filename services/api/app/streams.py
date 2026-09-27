"""Feeding ``FleetState`` from the telemetry and stop_events streams.

After a restart the state is rebuilt from the same streams, not from the
ingest and matcher tables: telemetry is read newest-first just past the
window, stop events are read whole (a few thousand a day). The live feed
then resumes from the newest id seen during recovery, so nothing is read
twice or skipped.
"""

import logging
from collections.abc import Awaitable, Callable

from pydantic import BaseModel, ValidationError
from redis.asyncio import Redis

from app.state import FleetState
from common.bus import STOP_EVENTS_STREAM, TELEMETRY_STREAM, read_stream
from contracts import StopEvent, TelemetryRecord

log = logging.getLogger(__name__)

PAGE = 1000


async def recover_telemetry(
    redis: Redis, state: FleetState, *, stream: str = TELEMETRY_STREAM, page: int = PAGE
) -> str:
    """Reload the last telemetry window into ``state``; returns the id to resume from.

    Reads newest-first and stops at the first record older than the
    window before the clock. That record is kept: it shows a full window
    was observed, so the rebuilt state is not flagged as warming up.
    Returns ``"0"`` for an empty stream.
    """
    records: list[TelemetryRecord] = []
    newest_id: str | None = None
    clock = None
    upper = "+"
    done = False
    while not done:
        entries = await redis.xrevrange(stream, max=upper, min="-", count=page)
        if not entries:
            break
        for raw_id, fields in entries:
            entry_id = _decode(raw_id)
            newest_id = newest_id or entry_id
            upper = f"({entry_id}"
            record = _parse(TelemetryRecord, fields, entry_id, stream)
            if record is None:
                continue
            records.append(record)
            if clock is None or record.event_time > clock:
                clock = record.event_time
            if record.event_time < clock - state.window:
                done = True
                break
    for record in reversed(records):
        state.add_telemetry(record)
    return newest_id or "0"


async def recover_stop_events(
    redis: Redis, state: FleetState, *, stream: str = STOP_EVENTS_STREAM, page: int = PAGE
) -> str:
    """Reload every stop event into ``state``; returns the id to resume from, ``"0"`` if none."""
    last_id = "0"
    lower = "-"
    while True:
        entries = await redis.xrange(stream, min=lower, max="+", count=page)
        if not entries:
            break
        for raw_id, fields in entries:
            last_id = _decode(raw_id)
            event = _parse(StopEvent, fields, last_id, stream)
            if event is not None:
                state.add_stop_event(event)
        lower = f"({last_id}"
    return last_id


async def feed_telemetry(
    redis: Redis,
    state: FleetState,
    last_id: str,
    *,
    stream: str = TELEMETRY_STREAM,
    block_ms: int = 1000,
    on_entry: Callable[[str], None] | None = None,
) -> None:
    """Apply telemetry arriving after ``last_id`` until cancelled.

    ``on_entry`` gets the id of every entry read, malformed ones aside.
    """
    async for entry_id, record in read_stream(
        redis, stream, TelemetryRecord, last_id=last_id, block_ms=block_ms
    ):
        state.add_telemetry(record)
        if on_entry is not None:
            on_entry(entry_id)


async def feed_stop_events(
    redis: Redis,
    state: FleetState,
    last_id: str,
    *,
    stream: str = STOP_EVENTS_STREAM,
    block_ms: int = 1000,
    on_event: Callable[[StopEvent], Awaitable[None]] | None = None,
) -> None:
    """Apply stop events arriving after ``last_id`` until cancelled.

    ``on_event`` is awaited for every event after the state has it; what
    it raises ends the feed.
    """
    async for _entry_id, event in read_stream(
        redis, stream, StopEvent, last_id=last_id, block_ms=block_ms
    ):
        state.add_stop_event(event)
        if on_event is not None:
            await on_event(event)


def _parse[M: BaseModel](model: type[M], fields: dict, entry_id: str, stream: str) -> M | None:
    raw = fields.get(b"data", fields.get("data"))
    try:
        return model.model_validate_json(raw)
    except ValidationError:
        log.warning("dropping malformed entry %s on %s: %.200r", entry_id, stream, raw)
        return None


def _decode(value: bytes | str) -> str:
    return value.decode() if isinstance(value, bytes) else value
