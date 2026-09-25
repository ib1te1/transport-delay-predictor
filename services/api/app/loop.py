"""The prediction loop: rebuild state, follow the streams, score on dataset-clock ticks.

Runs as one background task of the api process. Any failure short of
cancellation — Redis or Postgres gone, a bug inside a tick — is logged,
and the whole loop starts over after a doubling pause. Starting over
rebuilds the state from the streams, so it loses nothing they still hold.
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta

from psycopg_pool import ConnectionPool
from redis.asyncio import Redis

from app.config import ApiConfig
from app.models import PredictionRow
from app.planner import PlanIndex
from app.predictor_client import PredictorClient
from app.scoring import run_tick
from app.state import FleetState
from app.store import load_plan, save_predictions
from app.streams import feed_stop_events, feed_telemetry, recover_stop_events, recover_telemetry
from common.bus import PREDICTIONS_STREAM, STOP_EVENTS_STREAM, TELEMETRY_STREAM, append_async

log = logging.getLogger(__name__)

# How often, in real seconds, the dataset clock is checked for a due tick.
# Several checks a second keep up with accelerated replay.
CLOCK_POLL_SEC = 0.25

type Sleep = Callable[[float], Awaitable[None]]


@dataclass(frozen=True)
class StreamNames:
    """Streams the loop reads and writes; tests substitute their own."""

    telemetry: str = TELEMETRY_STREAM
    stop_events: str = STOP_EVENTS_STREAM
    predictions: str = PREDICTIONS_STREAM


DEFAULT_STREAMS = StreamNames()


async def schedule_ticks(
    state,
    tick: Callable[[datetime], Awaitable[object]],
    period: timedelta,
    *,
    poll_sec: float = CLOCK_POLL_SEC,
    sleep: Sleep = asyncio.sleep,
) -> None:
    """Call ``tick(T)`` whenever the dataset clock is a full period past the last ``T``.

    ``T`` is the clock itself, not a period boundary. Periods the clock
    skipped over during accelerated replay are not caught up.
    """
    last: datetime | None = None
    while True:
        clock = state.clock
        if clock is not None and (last is None or clock - last >= period):
            last = clock
            await tick(clock)
        await sleep(poll_sec)


def _load_plan(pool: ConnectionPool) -> PlanIndex:
    with pool.connection() as conn:
        return load_plan(conn)


def _save(pool: ConnectionPool, rows: list[PredictionRow]) -> None:
    with pool.connection() as conn:
        save_predictions(conn, rows)


async def run_once(
    pool: ConnectionPool,
    redis: Redis,
    predictor: PredictorClient,
    config: ApiConfig,
    *,
    streams: StreamNames = DEFAULT_STREAMS,
) -> None:
    """Load the plan, rebuild the state, then follow the streams and score until a failure."""
    plan = await asyncio.to_thread(_load_plan, pool)
    state = FleetState(timedelta(seconds=config.request.telemetry_window_sec))
    telemetry_id = await recover_telemetry(redis, state, stream=streams.telemetry)
    stop_events_id = await recover_stop_events(redis, state, stream=streams.stop_events)
    log.info(
        "prediction loop started: %d planned vehicles, %d vehicles in the window, clock %s",
        len(plan),
        len(state.vehicles()),
        state.clock,
    )

    async def save(rows: list[PredictionRow]) -> None:
        await asyncio.to_thread(_save, pool, rows)

    async def publish(rows: list[PredictionRow]) -> None:
        for row in rows:
            await append_async(redis, streams.predictions, row)

    async def tick(t: datetime) -> None:
        await run_tick(
            t, state, plan, config, predict=predictor.predict, save=save, publish=publish
        )

    async with asyncio.TaskGroup() as group:
        group.create_task(feed_telemetry(redis, state, telemetry_id, stream=streams.telemetry))
        group.create_task(
            feed_stop_events(redis, state, stop_events_id, stream=streams.stop_events)
        )
        group.create_task(schedule_ticks(state, tick, timedelta(seconds=config.scoring_period_sec)))


async def run_prediction_loop(
    pool: ConnectionPool,
    redis_url: str,
    predictor: PredictorClient,
    config: ApiConfig,
    *,
    streams: StreamNames = DEFAULT_STREAMS,
    initial_delay: float = 1.0,
    max_delay: float = 30.0,
    sleep: Sleep = asyncio.sleep,
) -> None:
    """Keep ``run_once`` going until cancelled, restarting it after any failure."""
    loop = asyncio.get_running_loop()
    delay = initial_delay
    while True:
        started = loop.time()
        redis = Redis.from_url(redis_url)
        try:
            await run_once(pool, redis, predictor, config, streams=streams)
        except Exception as exc:
            log.warning(
                "prediction loop failed (%s: %s), restarting in %.1fs",
                type(exc).__name__,
                exc,
                delay,
                exc_info=True,
            )
        finally:
            await redis.aclose()
        if loop.time() - started > max_delay:
            delay = initial_delay
        await sleep(delay)
        delay = min(delay * 2, max_delay)
