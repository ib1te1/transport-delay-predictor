"""The prediction loop: rebuild state, follow the streams, score on dataset-clock ticks.

Runs as one background task of the api process. Any failure short of
cancellation — Redis or Postgres gone, a bug inside a tick — is logged,
and the whole loop starts over after a doubling pause. Starting over
rebuilds the state from the streams and the current predictions from the
``predictions`` table, so it loses nothing they still hold.

Each run also feeds the dashboard: it attaches its state to the
process's ``Dashboard``, publishes every prediction row and alert change
there, and refreshes the dashboard's view once a second.

Every arrival from ``stop_events`` checks the predictions made for that
stop, confirms its open alert and cancels the vehicle's open alerts for
stops planned before it. On start the run checks all arrivals in the
stream again, so those that came while api was down are not missed;
rows already checked stay as they are.
"""

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta

from psycopg_pool import ConnectionPool
from redis.asyncio import Redis

from app.config import ApiConfig
from app.dashboard import Dashboard
from app.live import LiveState
from app.metrics import LoopMetrics
from app.models import AlertRow, PredictionRow
from app.planner import PlanIndex
from app.predictor_client import PredictorClient
from app.scoring import run_tick
from app.state import FleetState
from app.store import (
    Accuracy,
    check_predictions,
    close_passed_alerts,
    confirm_alerts,
    load_accuracy,
    load_alerts,
    load_latest_predictions,
    load_plan,
    save_predictions,
    update_alerts,
)
from app.streams import feed_stop_events, feed_telemetry, recover_stop_events, recover_telemetry
from common.bus import PREDICTIONS_STREAM, STOP_EVENTS_STREAM, TELEMETRY_STREAM, append_async
from contracts import PredictRequest, PredictResponse, StopEvent

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


def _save(pool: ConnectionPool, rows: list[PredictionRow]) -> list[PredictionRow]:
    with pool.connection() as conn:
        return save_predictions(conn, rows)


def _load_latest(pool: ConnectionPool, since: datetime) -> list[PredictionRow]:
    with pool.connection() as conn:
        return load_latest_predictions(conn, since)


def _update_alerts(
    pool: ConnectionPool, rows: list[PredictionRow], segment_from: dict[str, int | None]
) -> list[AlertRow]:
    with pool.connection() as conn:
        return update_alerts(conn, rows, segment_from)


def _check_arrivals(
    pool: ConnectionPool, events: Sequence[StopEvent], *, always_measure: bool = False
) -> tuple[list[AlertRow], Accuracy | None]:
    """Check predictions, close the alerts the arrivals settle; the closed alerts and the
    new accuracy, or ``None`` for it if no prediction changed."""
    with pool.connection() as conn:
        changed = check_predictions(conn, events)
        closed = confirm_alerts(conn, events) + close_passed_alerts(conn, events)
        accuracy = load_accuracy(conn) if changed or always_measure else None
    return closed, accuracy


def _load_open_alerts(pool: ConnectionPool) -> list[AlertRow]:
    with pool.connection() as conn:
        return load_alerts(conn, "open")


async def run_once(
    pool: ConnectionPool,
    redis: Redis,
    predictor: PredictorClient,
    config: ApiConfig,
    *,
    dashboard: Dashboard,
    metrics: LoopMetrics,
    streams: StreamNames = DEFAULT_STREAMS,
) -> None:
    """Rebuild the state, then follow the streams, score and feed the dashboard until a failure."""
    plan = await asyncio.to_thread(_load_plan, pool)
    state = FleetState(timedelta(seconds=config.request.telemetry_window_sec))
    telemetry_id = await recover_telemetry(redis, state, stream=streams.telemetry)
    stop_events_id = await recover_stop_events(redis, state, stream=streams.stop_events)
    live = LiveState(state, plan)
    if state.clock is not None:
        # Predictions that ended meanwhile are dropped by the first refresh.
        for row in await asyncio.to_thread(_load_latest, pool, state.clock - state.window):
            live.current.put(row)
    confirmed, accuracy = await asyncio.to_thread(
        _check_arrivals, pool, state.all_stop_events(), always_measure=True
    )
    open_alerts = await asyncio.to_thread(_load_open_alerts, pool)
    dashboard.attach(live)
    dashboard.set_alerts(open_alerts)
    dashboard.set_accuracy(*accuracy)
    log.info(
        "prediction loop started: %d planned vehicles, %d vehicles in the window, "
        "%d current predictions, %d open alerts, %d checked predictions, clock %s",
        len(plan),
        len(state.vehicles()),
        len(live.current.vehicles()),
        len(open_alerts),
        accuracy.checked_predictions,
        state.clock,
    )

    async def publish_alerts(alerts: list[AlertRow]) -> None:
        try:
            await dashboard.publish_alerts(alerts)
        except Exception:
            log.exception("dashboard update failed for %d alert(s); scoring continues", len(alerts))

    await publish_alerts(confirmed)

    async def predict(requests: Sequence[PredictRequest]) -> list[PredictResponse]:
        started = time.perf_counter()
        ok = False
        try:
            answers = await predictor.predict(requests)
            ok = True
            return answers
        finally:
            metrics.predict_call((time.perf_counter() - started) * 1000, ok=ok)

    async def save(rows: list[PredictionRow]) -> list[PredictionRow]:
        return await asyncio.to_thread(_save, pool, rows)

    async def publish(rows: list[PredictionRow]) -> None:
        # Alerts follow the rows already saved, in a transaction of their own.
        metrics.rows_written(rows)
        for row in rows:
            await append_async(redis, streams.predictions, row)
        segment_from = {
            row.sample_id: state.last_passed_stop(row.tr_id, row.t)
            for row in rows
            if row.risk_level == "red"
        }
        alerts = await asyncio.to_thread(_update_alerts, pool, rows, segment_from)
        try:
            await dashboard.publish_predictions(rows)
        except Exception:
            log.exception(
                "dashboard update failed for %d prediction row(s); scoring continues", len(rows)
            )
        await publish_alerts(alerts)

    async def tick(t: datetime) -> None:
        started = time.perf_counter()
        rows = await run_tick(t, state, plan, config, predict=predict, save=save, publish=publish)
        if rows:
            metrics.tick((time.perf_counter() - started) * 1000)

    async def arrived(event: StopEvent) -> None:
        confirmed, accuracy = await asyncio.to_thread(_check_arrivals, pool, [event])
        if accuracy is not None:
            dashboard.set_accuracy(*accuracy)
        await publish_alerts(confirmed)

    async with asyncio.TaskGroup() as group:
        group.create_task(
            feed_telemetry(
                redis,
                state,
                telemetry_id,
                stream=streams.telemetry,
                on_entry=metrics.telemetry_read,
            )
        )
        group.create_task(
            feed_stop_events(
                redis, state, stop_events_id, stream=streams.stop_events, on_event=arrived
            )
        )
        group.create_task(schedule_ticks(state, tick, timedelta(seconds=config.scoring_period_sec)))
        group.create_task(dashboard.run())


def _leaf_exceptions(exc: BaseExceptionGroup) -> list[BaseException]:
    """The non-group exceptions nested anywhere inside ``exc``, depth-first."""
    leaves: list[BaseException] = []
    for sub in exc.exceptions:
        if isinstance(sub, BaseExceptionGroup):
            leaves.extend(_leaf_exceptions(sub))
        else:
            leaves.append(sub)
    return leaves


def _failure_detail(exc: BaseException) -> str:
    """A loggable description of ``exc``; an ``ExceptionGroup`` expands to its leaves.

    The default ``str()`` of an ``ExceptionGroup`` raised out of a
    ``TaskGroup`` is just "unhandled errors in a TaskGroup" — the actual
    causes are nested inside and otherwise show up only in the traceback.
    """
    if isinstance(exc, BaseExceptionGroup):
        return "; ".join(repr(leaf) for leaf in _leaf_exceptions(exc))
    return str(exc)


async def run_prediction_loop(
    pool: ConnectionPool,
    redis_url: str,
    predictor: PredictorClient,
    config: ApiConfig,
    *,
    dashboard: Dashboard,
    metrics: LoopMetrics | None = None,
    streams: StreamNames = DEFAULT_STREAMS,
    initial_delay: float = 1.0,
    max_delay: float = 30.0,
    sleep: Sleep = asyncio.sleep,
) -> None:
    """Keep ``run_once`` going until cancelled, restarting it after any failure."""
    metrics = metrics or LoopMetrics()
    loop = asyncio.get_running_loop()
    delay = initial_delay
    while True:
        started = loop.time()
        redis: Redis | None = None
        try:
            redis = Redis.from_url(redis_url)
            await run_once(
                pool,
                redis,
                predictor,
                config,
                dashboard=dashboard,
                metrics=metrics,
                streams=streams,
            )
        except Exception as exc:
            log.warning(
                "prediction loop failed (%s: %s), restarting in %.1fs",
                type(exc).__name__,
                _failure_detail(exc),
                delay,
                exc_info=True,
            )
        finally:
            if redis is not None:
                await redis.aclose()
        if loop.time() - started > max_delay:
            delay = initial_delay
        await sleep(delay)
        delay = min(delay * 2, max_delay)
