import asyncio
import json
import logging
import random
from contextlib import suppress
from datetime import datetime, timedelta
from uuid import uuid4

import httpx
import pytest
from factories import at, plan_stop, prediction_row, stop_event, telemetry_record
from redis.asyncio import Redis

import app.loop as loop_module
from app.config import ApiConfig
from app.dashboard import Dashboard
from app.loop import StreamNames, run_prediction_loop, schedule_ticks
from app.metrics import LoopMetrics
from app.models import AlertRow, PredictionRow
from app.predictor_client import PredictorClient
from app.store import save_predictions, update_alerts
from common.bus import append_async
from common.db import connect, fetch_models, insert_models, make_pool


class Stop(Exception):
    pass


class ClockOnly:
    """Stands in for FleetState: schedule_ticks reads only ``clock``."""

    clock: datetime | None = None


@pytest.mark.anyio
async def test_ticks_follow_the_dataset_clock_and_skip_missed_periods() -> None:
    clocks = iter([None, at(0), at(30), at(59), at(60), at(61), at(500), at(530), at(560)])
    state = ClockOnly()
    ticks: list[datetime] = []

    async def tick(t: datetime) -> None:
        ticks.append(t)

    async def sleep(_: float) -> None:
        try:
            state.clock = next(clocks)
        except StopIteration:
            raise Stop from None

    with pytest.raises(Stop):
        await schedule_ticks(state, tick, timedelta(seconds=60), sleep=sleep)

    assert ticks == [at(0), at(60), at(500), at(560)]


@pytest.mark.anyio
async def test_loop_restarts_after_a_failure_with_a_doubling_pause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = 0

    async def failing(*args, **kwargs) -> None:
        nonlocal attempts
        attempts += 1
        raise ConnectionError("redis down")

    monkeypatch.setattr(loop_module, "run_once", failing)
    delays: list[float] = []

    async def sleep(delay: float) -> None:
        delays.append(delay)
        if len(delays) == 6:
            raise Stop

    with pytest.raises(Stop):
        await run_prediction_loop(
            None, "redis://unused:6379/0", None, ApiConfig(), dashboard=None, sleep=sleep
        )

    assert delays == [1.0, 2.0, 4.0, 8.0, 16.0, 30.0]
    assert attempts == 6


@pytest.mark.anyio
async def test_loop_retries_when_the_redis_client_cannot_be_created(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeRedis:
        async def aclose(self) -> None:
            pass

    creation_attempts = 0

    def failing_from_url(url: str) -> FakeRedis:
        nonlocal creation_attempts
        creation_attempts += 1
        if creation_attempts == 1:
            raise ValueError("malformed redis_url")
        return FakeRedis()

    monkeypatch.setattr(loop_module.Redis, "from_url", failing_from_url)

    run_once_calls = 0

    async def failing_run_once(*args, **kwargs) -> None:
        nonlocal run_once_calls
        run_once_calls += 1
        raise ConnectionError("redis down")

    monkeypatch.setattr(loop_module, "run_once", failing_run_once)
    delays: list[float] = []

    async def sleep(delay: float) -> None:
        delays.append(delay)
        if len(delays) == 3:
            raise Stop

    with pytest.raises(Stop):
        await run_prediction_loop(
            None, "redis://unused:6379/0", None, ApiConfig(), dashboard=None, sleep=sleep
        )

    # First iteration fails before run_once is ever called; it is retried
    # like any other failure instead of killing the supervisor.
    assert creation_attempts == 3
    assert run_once_calls == 2
    assert delays == [1.0, 2.0, 4.0]


@pytest.mark.anyio
async def test_loop_logs_the_leaf_exceptions_of_an_exception_group(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    async def failing(*args, **kwargs) -> None:
        raise ExceptionGroup(
            "unhandled errors in a TaskGroup",
            [ValueError("bad telemetry record"), RuntimeError("stop_events feed died")],
        )

    monkeypatch.setattr(loop_module, "run_once", failing)
    delays: list[float] = []

    async def sleep(delay: float) -> None:
        delays.append(delay)
        raise Stop

    with caplog.at_level(logging.WARNING, logger="app.loop"), pytest.raises(Stop):
        await run_prediction_loop(
            None, "redis://unused:6379/0", None, ApiConfig(), dashboard=None, sleep=sleep
        )

    message = caplog.records[-1].getMessage()
    assert "bad telemetry record" in message
    assert "stop_events feed died" in message


def fetch_rows(database_url: str, tr_id: int) -> list[PredictionRow]:
    with connect(database_url) as conn:
        return fetch_models(
            conn, PredictionRow, "SELECT * FROM predictions WHERE tr_id = %s ORDER BY t", (tr_id,)
        )


async def wait_for_rows(database_url: str, tr_id: int, count: int) -> list[PredictionRow]:
    async with asyncio.timeout(20):
        while len(rows := await asyncio.to_thread(fetch_rows, database_url, tr_id)) < count:
            await asyncio.sleep(0.1)
    return rows


async def wait_for_shown_prediction(dashboard: Dashboard, tr_id: int, sample_id: str) -> None:
    """Wait until the dashboard's published view shows ``sample_id`` for the vehicle."""

    def shown() -> bool:
        return any(
            v.tr_id == tr_id and v.prediction is not None and v.prediction.sample_id == sample_id
            for v in dashboard.snapshot().vehicles
        )

    async with asyncio.timeout(10):
        while not shown():
            await asyncio.sleep(0.1)


async def wait_for_shown_vehicle(dashboard: Dashboard, tr_id: int):
    """Wait until the dashboard's published view includes the vehicle; return its view."""

    def find():
        return next((v for v in dashboard.snapshot().vehicles if v.tr_id == tr_id), None)

    async with asyncio.timeout(10):
        while (vehicle := find()) is None:
            await asyncio.sleep(0.1)
        return vehicle


async def wait_for_stream_entries(redis: Redis, stream: str, count: int) -> list:
    async with asyncio.timeout(20):
        while len(entries := await redis.xrange(stream)) < count:
            await asyncio.sleep(0.1)
    return entries


class FailingDashboard(Dashboard):
    """A dashboard whose ``publish_predictions`` always fails, to test that scoring survives it."""

    async def publish_predictions(self, rows: list[PredictionRow]) -> None:
        raise RuntimeError("dashboard boom")


@pytest.mark.anyio
@pytest.mark.timeout(60)
async def test_loop_scores_recovered_and_live_telemetry(database_url: str, redis_url: str) -> None:
    # Ids far above anything in the real seed data; rows are deleted below.
    tr_id = 9_000_000_000 + random.randrange(1_000_000)
    target = plan_stop(tr_id, tr_id, at(1800) + timedelta(minutes=12))
    streams = StreamNames(
        telemetry=f"test-{uuid4().hex}",
        stop_events=f"test-{uuid4().hex}",
        predictions=f"test-{uuid4().hex}",
    )
    batches: list[list[dict]] = []

    def model(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        batches.append(body)
        return httpx.Response(
            200,
            json=[
                {
                    "sample_id": r["sample_id"],
                    "prediction_s": 400.0,
                    "p_late": None,
                    "reasons": ["accumulated_delay"],
                    "model_version": "m1",
                }
                for r in body
            ],
        )

    predictor = PredictorClient("http://predictor", 1.0, transport=httpx.MockTransport(model))
    redis = Redis.from_url(redis_url)
    dashboard = Dashboard(redis, ApiConfig(), channel=f"test-{uuid4().hex}")
    with connect(database_url) as conn:
        insert_models(conn, "stops_plan", [target])
        conn.commit()
    try:
        # Already in the streams before start: picked up by recovery.
        await append_async(redis, streams.telemetry, telemetry_record(tr_id, at(0)))
        await append_async(redis, streams.telemetry, telemetry_record(tr_id, at(1800)))
        passed_plan = at(1500)
        await append_async(
            redis,
            streams.stop_events,
            stop_event(tr_id, tr_id + 1, passed_plan, passed_plan + timedelta(seconds=90)),
        )
        with make_pool(database_url) as pool:
            task = asyncio.create_task(
                run_prediction_loop(
                    pool, redis_url, predictor, ApiConfig(), dashboard=dashboard, streams=streams
                )
            )
            try:
                await wait_for_rows(database_url, tr_id, 1)
                # Arrives while running: picked up by the live feed, next tick.
                await append_async(redis, streams.telemetry, telemetry_record(tr_id, at(1860)))
                rows = await wait_for_rows(database_url, tr_id, 2)
                await wait_for_shown_prediction(dashboard, tr_id, rows[-1].sample_id)
            finally:
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
        published = await redis.xrange(streams.predictions)
    finally:
        await redis.delete(streams.telemetry, streams.stop_events, streams.predictions)
        await redis.aclose()
        await predictor.aclose()
        with connect(database_url) as conn:
            conn.execute("DELETE FROM predictions WHERE tr_id = %s", (tr_id,))
            conn.execute("DELETE FROM alerts WHERE tr_id = %s", (tr_id,))
            conn.execute("DELETE FROM stops_plan WHERE stop_id = %s", (target.stop_id,))
            conn.commit()

    assert [r.t for r in rows] == [at(1800), at(1860)]
    first = rows[0]
    assert (first.target_stop_id, first.cur_dev_s, first.prediction_s) == (tr_id, 90.0, 400.0)
    assert (first.risk_level, first.degraded, first.model_version) == ("red", False, "m1")
    assert len(batches[0][0]["telemetry"]) == 2
    assert [PredictionRow.model_validate_json(f[b"data"]) for _, f in published] == rows


@pytest.mark.anyio
@pytest.mark.timeout(60)
async def test_loop_keeps_scoring_when_the_dashboard_update_fails(
    database_url: str, redis_url: str, caplog: pytest.LogCaptureFixture
) -> None:
    tr_id = 9_000_000_000 + random.randrange(1_000_000)
    target = plan_stop(tr_id, tr_id, at(1800) + timedelta(minutes=12))
    streams = StreamNames(
        telemetry=f"test-{uuid4().hex}",
        stop_events=f"test-{uuid4().hex}",
        predictions=f"test-{uuid4().hex}",
    )

    def model(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        return httpx.Response(
            200,
            json=[
                {
                    "sample_id": r["sample_id"],
                    "prediction_s": 400.0,
                    "p_late": None,
                    "reasons": ["accumulated_delay"],
                    "model_version": "m1",
                }
                for r in body
            ],
        )

    predictor = PredictorClient("http://predictor", 1.0, transport=httpx.MockTransport(model))
    redis = Redis.from_url(redis_url)
    dashboard = FailingDashboard(redis, ApiConfig(), channel=f"test-{uuid4().hex}")
    with connect(database_url) as conn:
        insert_models(conn, "stops_plan", [target])
        conn.commit()
    try:
        await append_async(redis, streams.telemetry, telemetry_record(tr_id, at(0)))
        await append_async(redis, streams.telemetry, telemetry_record(tr_id, at(1800)))
        with make_pool(database_url) as pool:
            with caplog.at_level(logging.WARNING, logger="app.loop"):
                task = asyncio.create_task(
                    run_prediction_loop(
                        pool,
                        redis_url,
                        predictor,
                        ApiConfig(),
                        dashboard=dashboard,
                        streams=streams,
                    )
                )
                try:
                    rows = await wait_for_rows(database_url, tr_id, 1)
                    published = await wait_for_stream_entries(redis, streams.predictions, 1)
                    assert not task.done()
                finally:
                    task.cancel()
                    with suppress(asyncio.CancelledError):
                        await task
    finally:
        await redis.delete(streams.telemetry, streams.stop_events, streams.predictions)
        await redis.aclose()
        await predictor.aclose()
        with connect(database_url) as conn:
            conn.execute("DELETE FROM predictions WHERE tr_id = %s", (tr_id,))
            conn.execute("DELETE FROM alerts WHERE tr_id = %s", (tr_id,))
            conn.execute("DELETE FROM stops_plan WHERE stop_id = %s", (target.stop_id,))
            conn.commit()

    assert rows[0].tr_id == tr_id
    assert [PredictionRow.model_validate_json(f[b"data"]) for _, f in published] == rows
    messages = [r.getMessage() for r in caplog.records]
    assert any(
        "dashboard update failed for 1 prediction row(s); scoring continues" in m for m in messages
    )
    assert not any("restarting" in m for m in messages)


@pytest.mark.anyio
@pytest.mark.timeout(60)
async def test_loop_restores_current_predictions_from_the_table(
    database_url: str, redis_url: str
) -> None:
    tr_id = 9_000_000_000 + random.randrange(1_000_000)
    streams = StreamNames(
        telemetry=f"test-{uuid4().hex}",
        stop_events=f"test-{uuid4().hex}",
        predictions=f"test-{uuid4().hex}",
    )
    stored = [
        prediction_row(sample_id=f"{tr_id}_a", tr_id=tr_id, t=at(1500), target_stop_id=tr_id),
        prediction_row(sample_id=f"{tr_id}_b", tr_id=tr_id, t=at(1700), target_stop_id=tr_id),
    ]
    # No predictor_url: nothing is planned for this vehicle, so no tick calls it.
    predictor = PredictorClient(None, 1.0)
    redis = Redis.from_url(redis_url)
    dashboard = Dashboard(redis, ApiConfig(), channel=f"test-{uuid4().hex}")
    with connect(database_url) as conn:
        save_predictions(conn, stored)
        conn.commit()
    try:
        await append_async(redis, streams.telemetry, telemetry_record(tr_id, at(0)))
        await append_async(redis, streams.telemetry, telemetry_record(tr_id, at(1800)))
        with make_pool(database_url) as pool:
            task = asyncio.create_task(
                run_prediction_loop(
                    pool, redis_url, predictor, ApiConfig(), dashboard=dashboard, streams=streams
                )
            )
            try:
                await wait_for_shown_prediction(dashboard, tr_id, f"{tr_id}_b")
            finally:
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
    finally:
        await redis.delete(streams.telemetry, streams.stop_events, streams.predictions)
        await redis.aclose()
        with connect(database_url) as conn:
            conn.execute("DELETE FROM predictions WHERE tr_id = %s", (tr_id,))
            conn.execute("DELETE FROM alerts WHERE tr_id = %s", (tr_id,))
            conn.commit()


@pytest.mark.anyio
@pytest.mark.timeout(60)
async def test_loop_restart_with_an_already_passed_target_shows_no_prediction(
    database_url: str, redis_url: str
) -> None:
    tr_id = 9_000_000_000 + random.randrange(1_000_000)
    streams = StreamNames(
        telemetry=f"test-{uuid4().hex}",
        stop_events=f"test-{uuid4().hex}",
        predictions=f"test-{uuid4().hex}",
    )
    stored = [
        prediction_row(sample_id=f"{tr_id}_a", tr_id=tr_id, t=at(1500), target_stop_id=tr_id),
        prediction_row(sample_id=f"{tr_id}_b", tr_id=tr_id, t=at(1700), target_stop_id=tr_id),
    ]
    # No predictor_url: nothing is planned for this vehicle, so no tick calls it.
    predictor = PredictorClient(None, 1.0)
    redis = Redis.from_url(redis_url)
    dashboard = Dashboard(redis, ApiConfig(), channel=f"test-{uuid4().hex}")
    with connect(database_url) as conn:
        save_predictions(conn, stored)
        conn.commit()
    try:
        await append_async(redis, streams.telemetry, telemetry_record(tr_id, at(0)))
        await append_async(redis, streams.telemetry, telemetry_record(tr_id, at(1800)))
        passed = at(1700)
        await append_async(
            redis,
            streams.stop_events,
            stop_event(tr_id, tr_id, passed, passed + timedelta(seconds=30)),
        )
        with make_pool(database_url) as pool:
            task = asyncio.create_task(
                run_prediction_loop(
                    pool, redis_url, predictor, ApiConfig(), dashboard=dashboard, streams=streams
                )
            )
            try:
                vehicle = await wait_for_shown_vehicle(dashboard, tr_id)
            finally:
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
    finally:
        await redis.delete(streams.telemetry, streams.stop_events, streams.predictions)
        await redis.aclose()
        with connect(database_url) as conn:
            conn.execute("DELETE FROM predictions WHERE tr_id = %s", (tr_id,))
            conn.execute("DELETE FROM alerts WHERE tr_id = %s", (tr_id,))
            conn.commit()

    assert vehicle.prediction is None


def fetch_alerts(database_url: str, tr_id: int) -> list[AlertRow]:
    with connect(database_url) as conn:
        return fetch_models(
            conn, AlertRow, "SELECT * FROM alerts WHERE tr_id = %s ORDER BY id", (tr_id,)
        )


async def wait_for_confirmed(database_url: str, tr_id: int) -> list[AlertRow]:
    async with asyncio.timeout(20):
        while True:
            alerts = await asyncio.to_thread(fetch_alerts, database_url, tr_id)
            if any(a.status == "confirmed" for a in alerts):
                return alerts
            await asyncio.sleep(0.1)


def shown_alerts(dashboard: Dashboard, tr_id: int) -> list:
    return [a for a in dashboard.snapshot().alerts if a.tr_id == tr_id]


def always_late(request: httpx.Request) -> httpx.Response:
    """predictor answering 400 s of delay to every request: red."""
    return httpx.Response(
        200,
        json=[
            {
                "sample_id": r["sample_id"],
                "prediction_s": 400.0,
                "p_late": 0.9,
                "reasons": ["accumulated_delay"],
                "model_version": "m1",
            }
            for r in json.loads(request.content)
        ],
    )


@pytest.mark.anyio
@pytest.mark.timeout(60)
async def test_a_red_prediction_opens_an_alert_the_arrival_confirms(
    database_url: str, redis_url: str
) -> None:
    tr_id = 9_000_000_000 + random.randrange(1_000_000)
    target = plan_stop(tr_id, tr_id, at(1800) + timedelta(minutes=12))
    streams = StreamNames(
        telemetry=f"test-{uuid4().hex}",
        stop_events=f"test-{uuid4().hex}",
        predictions=f"test-{uuid4().hex}",
    )
    predictor = PredictorClient("http://predictor", 1.0, transport=httpx.MockTransport(always_late))
    redis = Redis.from_url(redis_url)
    dashboard = Dashboard(redis, ApiConfig(), channel=f"test-{uuid4().hex}")
    metrics = LoopMetrics()
    with connect(database_url) as conn:
        insert_models(conn, "stops_plan", [target])
        conn.commit()
    try:
        await append_async(redis, streams.telemetry, telemetry_record(tr_id, at(0)))
        await append_async(redis, streams.telemetry, telemetry_record(tr_id, at(1800)))
        passed_plan = at(1500)
        await append_async(
            redis,
            streams.stop_events,
            stop_event(tr_id, tr_id + 1, passed_plan, passed_plan + timedelta(seconds=90)),
        )
        with make_pool(database_url) as pool:
            task = asyncio.create_task(
                run_prediction_loop(
                    pool,
                    redis_url,
                    predictor,
                    ApiConfig(),
                    dashboard=dashboard,
                    metrics=metrics,
                    streams=streams,
                )
            )
            try:
                await wait_for_rows(database_url, tr_id, 1)
                async with asyncio.timeout(10):
                    while not shown_alerts(dashboard, tr_id):
                        await asyncio.sleep(0.1)
                # 400 s late, as predicted
                arrival_time = target.time_plan + timedelta(seconds=400)
                await append_async(
                    redis,
                    streams.stop_events,
                    stop_event(tr_id, target.stop_id, target.time_plan, arrival_time),
                )
                [alert] = await wait_for_confirmed(database_url, tr_id)
                async with asyncio.timeout(10):
                    while shown_alerts(dashboard, tr_id):
                        await asyncio.sleep(0.1)
                rows = await asyncio.to_thread(fetch_rows, database_url, tr_id)
            finally:
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
    finally:
        await redis.delete(streams.telemetry, streams.stop_events, streams.predictions)
        await redis.aclose()
        await predictor.aclose()
        with connect(database_url) as conn:
            conn.execute("DELETE FROM predictions WHERE tr_id = %s", (tr_id,))
            conn.execute("DELETE FROM alerts WHERE tr_id = %s", (tr_id,))
            conn.execute("DELETE FROM stops_plan WHERE stop_id = %s", (target.stop_id,))
            conn.commit()

    [row] = rows
    assert (row.actual_delay_s, row.abs_error_s) == (400.0, 0.0)
    assert (alert.opened_at, alert.segment_from_stop_id) == (at(1800), tr_id + 1)
    assert (alert.closed_at, alert.actual_delay_s) == (arrival_time, 400.0)
    # opened 12 min before the planned arrival, which came 400 s late
    assert alert.lead_time_s == 720.0 + 400.0
    assert dashboard.snapshot().checked_predictions >= 1
    assert metrics.predict_latency_ms().count == 1
    assert metrics.scoring_tick_ms().count == 1


@pytest.mark.anyio
@pytest.mark.timeout(60)
async def test_arrivals_that_came_while_api_was_down_are_checked_on_start(
    database_url: str, redis_url: str
) -> None:
    tr_id = 9_000_000_000 + random.randrange(1_000_000)
    streams = StreamNames(
        telemetry=f"test-{uuid4().hex}",
        stop_events=f"test-{uuid4().hex}",
        predictions=f"test-{uuid4().hex}",
    )
    stored = prediction_row(
        sample_id=f"{tr_id}_a",
        tr_id=tr_id,
        t=at(1500),
        target_stop_id=tr_id,
        prediction_s=400.0,
        risk_level="red",
    )
    # No predictor_url: nothing is planned for this vehicle, so no tick calls it.
    predictor = PredictorClient(None, 1.0)
    redis = Redis.from_url(redis_url)
    dashboard = Dashboard(redis, ApiConfig(), channel=f"test-{uuid4().hex}")
    with connect(database_url) as conn:
        save_predictions(conn, [stored])
        update_alerts(conn, [stored], {})
        conn.commit()
    try:
        await append_async(redis, streams.telemetry, telemetry_record(tr_id, at(0)))
        await append_async(redis, streams.telemetry, telemetry_record(tr_id, at(1800)))
        planned = at(1500) + timedelta(minutes=12)
        await append_async(
            redis,
            streams.stop_events,
            stop_event(tr_id, tr_id, planned, planned + timedelta(seconds=300)),
        )
        with make_pool(database_url) as pool:
            task = asyncio.create_task(
                run_prediction_loop(
                    pool, redis_url, predictor, ApiConfig(), dashboard=dashboard, streams=streams
                )
            )
            try:
                [alert] = await wait_for_confirmed(database_url, tr_id)
                await wait_for_shown_vehicle(dashboard, tr_id)
                [row] = await asyncio.to_thread(fetch_rows, database_url, tr_id)
            finally:
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
    finally:
        await redis.delete(streams.telemetry, streams.stop_events, streams.predictions)
        await redis.aclose()
        with connect(database_url) as conn:
            conn.execute("DELETE FROM predictions WHERE tr_id = %s", (tr_id,))
            conn.execute("DELETE FROM alerts WHERE tr_id = %s", (tr_id,))
            conn.commit()

    assert (row.actual_delay_s, row.abs_error_s) == (300.0, 100.0)
    assert (alert.status, alert.lead_time_s) == ("confirmed", 720.0 + 300.0)
    assert shown_alerts(dashboard, tr_id) == []


@pytest.mark.anyio
@pytest.mark.timeout(60)
async def test_an_arrival_past_the_target_cancels_the_alert_and_clears_it_from_the_dashboard(
    database_url: str, redis_url: str
) -> None:
    tr_id = 9_000_000_000 + random.randrange(1_000_000)
    streams = StreamNames(
        telemetry=f"test-{uuid4().hex}",
        stop_events=f"test-{uuid4().hex}",
        predictions=f"test-{uuid4().hex}",
    )
    target = plan_stop(tr_id, tr_id, at(1500) + timedelta(minutes=12))
    stored = prediction_row(
        sample_id=f"{tr_id}_a",
        tr_id=tr_id,
        t=at(1500),
        target_stop_id=target.stop_id,
        prediction_s=400.0,
        risk_level="red",
    )
    # No predictor_url: nothing is planned for this vehicle, so no tick calls it.
    predictor = PredictorClient(None, 1.0)
    redis = Redis.from_url(redis_url)
    dashboard = Dashboard(redis, ApiConfig(), channel=f"test-{uuid4().hex}")
    with connect(database_url) as conn:
        insert_models(conn, "stops_plan", [target])
        save_predictions(conn, [stored])
        update_alerts(conn, [stored], {})
        conn.commit()
    try:
        await append_async(redis, streams.telemetry, telemetry_record(tr_id, at(0)))
        await append_async(redis, streams.telemetry, telemetry_record(tr_id, at(1800)))
        # no arrival at the target; the next stop comes a minute after it in the plan
        passed_plan = target.time_plan + timedelta(minutes=1)
        passed_at = passed_plan + timedelta(seconds=300)
        await append_async(
            redis,
            streams.stop_events,
            stop_event(tr_id, tr_id + 1, passed_plan, passed_at),
        )
        with make_pool(database_url) as pool:
            task = asyncio.create_task(
                run_prediction_loop(
                    pool, redis_url, predictor, ApiConfig(), dashboard=dashboard, streams=streams
                )
            )
            try:
                async with asyncio.timeout(20):
                    while True:
                        alerts = await asyncio.to_thread(fetch_alerts, database_url, tr_id)
                        if all(a.status != "open" for a in alerts):
                            break
                        await asyncio.sleep(0.1)
                await wait_for_shown_vehicle(dashboard, tr_id)
            finally:
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
    finally:
        await redis.delete(streams.telemetry, streams.stop_events, streams.predictions)
        await redis.aclose()
        with connect(database_url) as conn:
            conn.execute("DELETE FROM predictions WHERE tr_id = %s", (tr_id,))
            conn.execute("DELETE FROM alerts WHERE tr_id = %s", (tr_id,))
            conn.execute("DELETE FROM stops_plan WHERE stop_id = %s", (target.stop_id,))
            conn.commit()

    [alert] = alerts
    assert (alert.status, alert.closed_at) == ("cancelled", passed_at)
    assert (alert.actual_delay_s, alert.lead_time_s) == (None, None)
    assert shown_alerts(dashboard, tr_id) == []
