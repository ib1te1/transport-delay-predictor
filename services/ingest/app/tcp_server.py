"""NDTP listener: reconnectable terminal sessions feeding the telemetry outbox."""

import asyncio
import logging
from collections import Counter
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from uuid import UUID, uuid4

from psycopg_pool import ConnectionPool
from redis.asyncio import Redis

from app.config import NdtpConfig
from app.ndtp import CellStop, NdtpError, parse_handshake, parse_realtime, read_frame
from app.normalize import from_ndtp
from app.outbox import Outbox
from app.store import load_vehicles, save_telemetry, set_emulator_shift, start_emulator_run
from contracts import TelemetryRecord

log = logging.getLogger(__name__)

type Save = Callable[[str, TelemetryRecord], Awaitable[None]]
type Shift = Callable[[int], Awaitable[float]]


def _vehicles(pool: ConnectionPool) -> dict[int, int]:
    with pool.connection() as conn:
        return load_vehicles(conn)


def _start(pool: ConnectionPool, period: str, anchor: datetime):
    with pool.connection() as conn:
        return start_emulator_run(conn, period=period, dataset_anchor=anchor)


def _set_shift(pool: ConnectionPool, run_id: UUID, shift_s: float) -> float:
    with pool.connection() as conn:
        return set_emulator_shift(conn, run_id, shift_s)


def _save(pool: ConnectionPool, run_id: UUID, source_key: str, record: TelemetryRecord) -> None:
    with pool.connection() as conn:
        save_telemetry(conn, run_id, source_key, record)


def _format_counts(counts: Counter) -> str:
    """One log-friendly line, e.g. ``unknown_cell type 99: 3`` or ``crc: 2``."""
    parts = []
    for key, count in sorted(counts.items()):
        label = f"{key[0]} type {key[1]}" if isinstance(key, tuple) else str(key)
        parts.append(f"{label}: {count}")
    return ", ".join(parts)


def _count_cell_stop(counts: Counter[tuple[str, int]], unit_id: int, stop: CellStop) -> None:
    """Log the first stop of each kind; the rest only go into the session summary."""
    key = (stop.reason, stop.cell_type)
    counts[key] += 1
    if counts[key] == 1:
        log.warning(
            "NDTP frame from %d: %s, cell type %d at offset %d; navigation kept,"
            " remaining cells skipped",
            unit_id,
            stop.reason,
            stop.cell_type,
            stop.offset,
        )


async def serve_client(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    config: NdtpConfig,
    vehicles: dict[int, int],
    *,
    shift_for: Shift,
    save: Save,
) -> None:
    """Accept one handshake, then valid realtime frames until disconnect."""
    session_id = uuid4()
    ordinal = 0
    unit_id: int | None = None
    cell_stops: Counter[tuple[str, int]] = Counter()
    try:
        first = await read_frame(
            reader, max_frame_bytes=config.max_frame_bytes, timeout_sec=config.read_timeout_sec
        )
        unit_id = parse_handshake(first)
        while True:
            try:
                frame = await read_frame(
                    reader,
                    max_frame_bytes=config.max_frame_bytes,
                    timeout_sec=config.read_timeout_sec,
                )
                ordinal += 1
                realtime = parse_realtime(frame, expected_unit_id=unit_id)
                if realtime.stop is not None:
                    _count_cell_stop(cell_stops, unit_id, realtime.stop)
                nav = realtime.nav
                shift_s = await shift_for(nav.timestamp)
                result = from_ndtp(
                    unit_id=unit_id,
                    timestamp=nav.timestamp,
                    longitude=nav.longitude,
                    latitude=nav.latitude,
                    extra_dop=nav.extra_dop,
                    speed_avg=nav.speed_avg,
                    course=nav.course,
                    timestamp_shift_s=shift_s,
                    vehicles=vehicles,
                )
                if result.record is None:
                    log.warning("dropping NDTP packet from %d: %s", unit_id, result.reason)
                    continue
                await save(f"{session_id}:{ordinal}", result.record)
            except NdtpError as exc:
                log.warning("dropping NDTP frame from %d: %s", unit_id, exc.reason)
                if exc.fatal:
                    return
    except (asyncio.IncompleteReadError, TimeoutError, ConnectionResetError):
        pass
    except NdtpError as exc:
        log.warning("rejecting NDTP connection: %s", exc.reason)
    except asyncio.CancelledError:
        raise
    except Exception:
        log.exception("NDTP connection failed; terminal may reconnect")
    finally:
        if cell_stops:
            log.info(
                "NDTP session from %s closed; auxiliary cells skipped: %s",
                unit_id,
                _format_counts(cell_stops),
            )
        writer.close()
        try:
            await writer.wait_closed()
        except ConnectionError:
            pass


async def run_ndtp_server(
    pool: ConnectionPool,
    redis_url: str,
    config: NdtpConfig,
    *,
    period: str,
    dataset_anchor: datetime,
) -> None:
    """Run the listener and its pending-row publisher until cancelled."""
    vehicles = await asyncio.to_thread(_vehicles, pool)
    run = await asyncio.to_thread(_start, pool, period, dataset_anchor.astimezone(UTC))
    redis = Redis.from_url(redis_url)
    outbox = Outbox(pool, redis)

    async def shift_for(timestamp: int) -> float:
        raw_shift = (dataset_anchor - datetime.fromtimestamp(timestamp, UTC)).total_seconds()
        return await asyncio.to_thread(_set_shift, pool, run.run_id, raw_shift)

    async def save(source_key: str, record: TelemetryRecord) -> None:
        await asyncio.to_thread(_save, pool, run.run_id, source_key, record)
        outbox.notify()

    async def handler(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        await serve_client(reader, writer, config, vehicles, shift_for=shift_for, save=save)

    try:
        server = await asyncio.start_server(handler, config.host, config.port)
        async with server, asyncio.TaskGroup() as group:
            log.info("NDTP listener ready on %s:%d", config.host, config.port)
            group.create_task(outbox.run())
            group.create_task(server.serve_forever())
    finally:
        await redis.aclose()
