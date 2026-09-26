"""Paced CSV playback through the same telemetry store and Redis outbox."""

import argparse
import asyncio
import hashlib
import logging
import shutil
import signal
import tempfile
import threading
import time
from collections import Counter
from collections.abc import Awaitable, Callable, Iterable, Sequence
from contextlib import closing
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from psycopg_pool import ConnectionPool
from redis.asyncio import Redis

from app.config import ReplayConfig, load_replay_config, to_utc
from app.normalize import from_csv
from app.outbox import Outbox
from app.sort import SortedCsvRow, SortResult, read_sorted, sort_to_file
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
type Save = Callable[[int, TelemetryRecord, tuple[datetime, int]], Awaitable[int]]
type Skip = Callable[[tuple[datetime, int]], Awaitable[None]]
type Delivered = Callable[[int], Awaitable[None]]
type Clock = Callable[[], float]

SORT_DIR_PREFIX = "ingest-replay-"
# docker compose stop escalates to SIGKILL after 10 s; the sort thread gets half.
SORT_STOP_WAIT_SEC = 5.0
LAG_LOG_THRESHOLD_SEC = 1.0
LAG_LOG_INTERVAL_SEC = 60.0


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


def remove_stale_sort_dirs(root: Path) -> None:
    """Delete sort directories left by a killed replay; one replay runs per container."""
    for stale in root.glob(f"{SORT_DIR_PREFIX}*"):
        if stale.is_dir():
            log.info("removing stale replay sort directory %s", stale)
            shutil.rmtree(stale, ignore_errors=True)


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


async def sort_in_thread(
    path: Path,
    source_zone: ZoneInfo,
    chunk_rows: int,
    out_dir: Path,
    *,
    start_at: datetime | None,
    end_at: datetime | None,
    after: tuple[datetime, int] | None,
) -> SortResult:
    """Run the external sort in a worker thread that stops when this task is cancelled.

    A thread cannot be cancelled, so cancellation sets ``stop`` and waits a
    bounded time for the thread to leave before the caller removes its files.
    """
    stop = threading.Event()
    worker = asyncio.create_task(
        asyncio.to_thread(
            sort_to_file,
            path,
            source_zone,
            chunk_rows,
            out_dir,
            start_at=start_at,
            end_at=end_at,
            after=after,
            stop=stop,
        )
    )
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError:
        stop.set()
        done, _ = await asyncio.wait({worker}, timeout=SORT_STOP_WAIT_SEC)
        if not done:
            log.warning("replay sort did not stop within %.0f s", SORT_STOP_WAIT_SEC)
        elif not worker.cancelled():
            # Expected SortCancelled; retrieving it keeps asyncio from reporting it.
            worker.exception()
        raise


async def play_sorted_rows(
    rows: Iterable[SortedCsvRow],
    config: ReplayConfig,
    source_zone: ZoneInfo,
    vehicles: dict[int, int],
    *,
    save: Save,
    skip: Skip,
    delivered: Delivered,
    sleep: Sleep = asyncio.sleep,
    clock: Clock = time.monotonic,
) -> ReplayStats:
    """Publish sorted rows on a schedule anchored at the first row this process publishes.

    Each row is due at anchor_wall + (event_time - anchor_event) / speedup,
    so the time spent saving and delivering shortens the next pause instead
    of adding up. A late row goes out at once; the anchor never moves, so
    playback returns to schedule after a stall.
    """
    stats = ReplayStats()
    anchor: tuple[float, datetime] | None = None
    last_lag_log: float | None = None
    for item in rows:
        result = from_csv(item.values, vehicles, source_zone)
        if result.record is None:
            reason = result.reason or "invalid"
            stats.skipped[reason] += 1
            if stats.skipped[reason] <= 20:
                log.warning("skipping CSV line %d: %s", item.line_number, reason)
            await skip(item.key)
            continue
        if anchor is None:
            anchor = (clock(), item.event_time)
        else:
            anchor_wall, anchor_event = anchor
            due = anchor_wall + (item.event_time - anchor_event).total_seconds() / config.speedup
            now = clock()
            delay = due - now
            if delay > 0:
                await sleep(delay)
            elif -delay > LAG_LOG_THRESHOLD_SEC and (
                last_lag_log is None or now - last_lag_log >= LAG_LOG_INTERVAL_SEC
            ):
                log.info("replay lag_sec=%.1f", -delay)
                last_lag_log = now
        row_id = await save(item.line_number, result.record, item.key)
        await delivered(row_id)
        stats.published += 1
    return stats


async def sort_and_play(
    path: Path,
    config: ReplayConfig,
    source_zone: ZoneInfo,
    vehicles: dict[int, int],
    work_dir: Path,
    *,
    after: tuple[datetime, int] | None,
    save: Save,
    skip: Skip,
    delivered: Delivered,
    sleep: Sleep = asyncio.sleep,
    clock: Clock = time.monotonic,
) -> ReplayStats:
    """Sort the selected range off the event loop, then play what follows ``after``."""
    start_at = to_utc(config.start_at, source_zone) if config.start_at else None
    end_at = to_utc(config.end_at, source_zone) if config.end_at else None
    result = await sort_in_thread(
        path,
        source_zone,
        config.sort_chunk_rows,
        work_dir,
        start_at=start_at,
        end_at=end_at,
        after=after,
    )
    if not result.selected:
        raise ValueError("the selected replay time range contains no telemetry")
    with closing(read_sorted(result.path)) as rows:
        stats = await play_sorted_rows(
            rows,
            config,
            source_zone,
            vehicles,
            save=save,
            skip=skip,
            delivered=delivered,
            sleep=sleep,
            clock=clock,
        )
    stats.read = result.read
    if result.bad_time:
        stats.skipped["event_time"] += result.bad_time
    return stats


async def run_replay(
    settings: ServiceSettings,
    *,
    new_run: bool = False,
    sleep: Sleep = asyncio.sleep,
    clock: Clock = time.monotonic,
) -> ReplayStats:
    """Resume an interrupted replay or run a new one when explicitly requested."""
    config, dataset = load_replay_config(settings.config_path)
    path = settings.data_dir / config.period / "traffic.csv"
    if not path.is_file():
        raise FileNotFoundError(f"replay dataset is missing: {path}")
    start_at = to_utc(config.start_at, dataset.zone) if config.start_at else None
    end_at = to_utc(config.end_at, dataset.zone) if config.end_at else None
    remove_stale_sort_dirs(Path(tempfile.gettempdir()))
    pool = make_pool(settings.database_url)
    pool.open(wait=False)
    redis = Redis.from_url(settings.redis_url)
    run: RunState | None = None
    completed = False
    work_dir: Path | None = None
    try:
        outbox = Outbox(pool, redis)
        await outbox.drain()
        run, vehicles = await asyncio.to_thread(
            _start, pool, path, config, start_at, end_at, new_run=new_run
        )
        work_dir = Path(tempfile.mkdtemp(prefix=SORT_DIR_PREFIX))

        async def save(
            line_number: int, record: TelemetryRecord, cursor: tuple[datetime, int]
        ) -> int:
            return await asyncio.to_thread(_save, pool, run, line_number, record, cursor)

        async def skip(cursor: tuple[datetime, int]) -> None:
            await asyncio.to_thread(_skip, pool, run, cursor)

        stats = await sort_and_play(
            path,
            config,
            dataset.zone,
            vehicles,
            work_dir,
            after=run.cursor,
            save=save,
            skip=skip,
            delivered=outbox.wait_for,
            sleep=sleep,
            clock=clock,
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
        if work_dir is not None:
            shutil.rmtree(work_dir, ignore_errors=True)
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
