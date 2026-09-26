"""Paced CSV playback through the same telemetry store and Redis outbox."""

import argparse
import asyncio
import hashlib
import logging
import signal
from collections import Counter
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from psycopg_pool import ConnectionPool
from redis.asyncio import Redis

from app.config import ReplayConfig, load_replay_config, to_utc
from app.normalize import from_csv
from app.outbox import Outbox
from app.sort import sorted_csv_rows
from app.store import (
    RunState,
    advance_replay_cursor,
    load_vehicles,
    save_telemetry,
    set_run_status,
    start_replay_run,
)
from common.config import ServiceSettings
from common.db import make_pool
from contracts import TelemetryRecord

log = logging.getLogger(__name__)

type Sleep = Callable[[float], Awaitable[None]]


@dataclass
class ReplayStats:
    read: int = 0
    published: int = 0
    skipped: Counter[str] = field(default_factory=Counter)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _start(
    pool: ConnectionPool,
    path: Path,
    config: ReplayConfig,
    start_at: datetime | None,
    end_at: datetime | None,
    *,
    new_run: bool,
) -> tuple[RunState, dict[int, int]]:
    with pool.connection() as conn:
        run = start_replay_run(
            conn,
            period=config.period,
            file_path=path,
            file_sha256=file_sha256(path),
            start_at=start_at,
            end_at=end_at,
            speedup=config.speedup,
            new_run=new_run,
        )
        return run, load_vehicles(conn)


def _save(
    pool: ConnectionPool,
    run: RunState,
    line_number: int,
    record: TelemetryRecord,
    cursor: tuple[datetime, int],
) -> int:
    with pool.connection() as conn:
        return save_telemetry(conn, run.run_id, str(line_number), record, replay_cursor=cursor)


def _skip(pool: ConnectionPool, run: RunState, cursor: tuple[datetime, int]) -> None:
    with pool.connection() as conn:
        advance_replay_cursor(conn, run.run_id, *cursor)


def _status(pool: ConnectionPool, run: RunState, status: str) -> None:
    with pool.connection() as conn:
        set_run_status(conn, run.run_id, status)


async def play_sorted_rows(
    path: Path,
    config: ReplayConfig,
    source_zone,
    vehicles: dict[int, int],
    run: RunState,
    *,
    save: Callable[[int, TelemetryRecord, tuple[datetime, int]], Awaitable[int]],
    skip: Callable[[tuple[datetime, int]], Awaitable[None]],
    delivered: Callable[[int], Awaitable[None]],
    sleep: Sleep = asyncio.sleep,
) -> ReplayStats:
    """Pace sorted points and keep the cursor at the last stored input."""
    stats = ReplayStats()

    def bad_time(line_number: int) -> None:
        stats.skipped["event_time"] += 1
        if stats.skipped["event_time"] <= 20:
            log.warning("skipping CSV line %d: invalid event_time", line_number)

    start_at = to_utc(config.start_at, source_zone) if config.start_at else None
    end_at = to_utc(config.end_at, source_zone) if config.end_at else None
    previous_published: datetime | None = None
    selected = 0

    def read_row() -> None:
        stats.read += 1

    for item in sorted_csv_rows(
        path, source_zone, config.sort_chunk_rows, on_bad_time=bad_time, on_row=read_row
    ):
        if start_at is not None and item.event_time < start_at:
            continue
        if end_at is not None and item.event_time > end_at:
            break
        selected += 1
        if run.cursor is not None and item.key <= run.cursor:
            continue
        result = from_csv(item.values, vehicles, source_zone)
        if result.record is None:
            reason = result.reason or "invalid"
            stats.skipped[reason] += 1
            if stats.skipped[reason] <= 20:
                log.warning("skipping CSV line %d: %s", item.line_number, reason)
            await skip(item.key)
            continue
        if previous_published is not None:
            delay = (item.event_time - previous_published).total_seconds() / config.speedup
            if delay > 0:
                await sleep(delay)
        row_id = await save(item.line_number, result.record, item.key)
        await delivered(row_id)
        stats.published += 1
        previous_published = item.event_time
    if not selected:
        raise ValueError("the selected replay time range contains no telemetry")
    return stats


async def run_replay(
    settings: ServiceSettings, *, new_run: bool = False, sleep: Sleep = asyncio.sleep
) -> ReplayStats:
    """Resume an interrupted replay or run a new one when explicitly requested."""
    config, dataset = load_replay_config(settings.config_path)
    path = settings.data_dir / config.period / "traffic.csv"
    if not path.is_file():
        raise FileNotFoundError(f"replay dataset is missing: {path}")
    start_at = to_utc(config.start_at, dataset.zone) if config.start_at else None
    end_at = to_utc(config.end_at, dataset.zone) if config.end_at else None
    pool = make_pool(settings.database_url)
    pool.open(wait=False)
    redis = Redis.from_url(settings.redis_url)
    run: RunState | None = None
    completed = False
    try:
        outbox = Outbox(pool, redis)
        await outbox.drain()
        run, vehicles = await asyncio.to_thread(
            _start, pool, path, config, start_at, end_at, new_run=new_run
        )

        async def save(
            line_number: int, record: TelemetryRecord, cursor: tuple[datetime, int]
        ) -> int:
            return await asyncio.to_thread(_save, pool, run, line_number, record, cursor)

        async def skip(cursor: tuple[datetime, int]) -> None:
            await asyncio.to_thread(_skip, pool, run, cursor)

        stats = await play_sorted_rows(
            path,
            config,
            dataset.zone,
            vehicles,
            run,
            save=save,
            skip=skip,
            delivered=outbox.wait_for,
            sleep=sleep,
        )
        await asyncio.to_thread(_status, pool, run, "completed")
        completed = True
        log.info(
            "replay complete: %d rows read, %d published, %d skipped: %s",
            stats.read,
            stats.published,
            sum(stats.skipped.values()),
            dict(stats.skipped),
        )
        return stats
    finally:
        if run is not None and not completed:
            try:
                await asyncio.to_thread(_status, pool, run, "interrupted")
            except Exception:
                log.exception("could not mark interrupted replay run")
        await redis.aclose()
        await asyncio.to_thread(pool.close)


async def _run_with_signals(settings: ServiceSettings, *, new_run: bool) -> ReplayStats:
    loop = asyncio.get_running_loop()
    task = asyncio.create_task(run_replay(settings, new_run=new_run))
    try:
        loop.add_signal_handler(signal.SIGTERM, task.cancel)
    except NotImplementedError:
        pass
    try:
        return await task
    finally:
        try:
            loop.remove_signal_handler(signal.SIGTERM)
        except NotImplementedError:
            pass


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Replay one dataset period into telemetry")
    parser.add_argument(
        "--new-run", action="store_true", help="start a new run after resetting demo state"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        asyncio.run(_run_with_signals(ServiceSettings(), new_run=args.new_run))
    except asyncio.CancelledError:
        log.info("replay interrupted")
        return 130
    except (OSError, ValueError) as exc:
        log.error("replay failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
