import asyncio
import json
import random
from contextlib import suppress
from datetime import datetime, timedelta
from uuid import uuid4

import httpx
import pytest
from factories import at, plan_stop, stop_event, telemetry_record
from redis.asyncio import Redis

import app.loop as loop_module
from app.config import ApiConfig
from app.loop import StreamNames, run_prediction_loop, schedule_ticks
from app.models import PredictionRow
from app.predictor_client import PredictorClient
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
        await run_prediction_loop(None, "redis://unused:6379/0", None, ApiConfig(), sleep=sleep)

    assert delays == [1.0, 2.0, 4.0, 8.0, 16.0, 30.0]
    assert attempts == 6


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
                run_prediction_loop(pool, redis_url, predictor, ApiConfig(), streams=streams)
            )
            try:
                await wait_for_rows(database_url, tr_id, 1)
                # Arrives while running: picked up by the live feed, next tick.
                await append_async(redis, streams.telemetry, telemetry_record(tr_id, at(1860)))
                rows = await wait_for_rows(database_url, tr_id, 2)
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
            conn.execute("DELETE FROM stops_plan WHERE stop_id = %s", (target.stop_id,))
            conn.commit()

    assert [r.t for r in rows] == [at(1800), at(1860)]
    first = rows[0]
    assert (first.target_stop_id, first.cur_dev_s, first.prediction_s) == (tr_id, 90.0, 400.0)
    assert (first.risk_level, first.degraded, first.model_version) == ("red", False, "m1")
    assert len(batches[0][0]["telemetry"]) == 2
    assert [PredictionRow.model_validate_json(f[b"data"]) for _, f in published] == rows
