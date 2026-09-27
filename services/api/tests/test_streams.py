import asyncio
from contextlib import suppress
from datetime import timedelta
from uuid import uuid4

import pytest
from factories import at, stop_event, telemetry_record
from redis.asyncio import Redis

from app.state import FleetState
from app.streams import (
    feed_stop_events,
    feed_telemetry,
    recover_stop_events,
    recover_telemetry,
)
from common.bus import append_async

WINDOW = timedelta(seconds=1800)


def stream_name() -> str:
    return f"test-{uuid4().hex}"


async def wait_until(condition, timeout: float = 10.0) -> None:
    async with asyncio.timeout(timeout):
        while not condition():
            await asyncio.sleep(0.02)


async def stop(*tasks: asyncio.Task) -> None:
    for task in tasks:
        task.cancel()
    for task in tasks:
        with suppress(asyncio.CancelledError):
            await task


@pytest.mark.anyio
@pytest.mark.timeout(30)
async def test_recover_telemetry_reads_back_to_one_record_past_the_window(redis_url: str) -> None:
    stream = stream_name()
    redis = Redis.from_url(redis_url)
    state = FleetState(WINDOW)
    try:
        for minute in range(51):
            newest = await append_async(redis, stream, telemetry_record(7, at(minute * 60)))
        resume_id = await recover_telemetry(redis, state, stream=stream, page=7)
    finally:
        await redis.delete(stream)
        await redis.aclose()

    assert resume_id == newest
    assert state.clock == at(3000)
    times = [p.event_time for p in state.telemetry(7)]
    # The window starts at at(1200); at(1140) is the first record older than it.
    assert times[0] == at(1140)
    assert times[-1] == at(3000)
    assert len(times) == 32
    assert not state.warming_up(at(3000))


@pytest.mark.anyio
@pytest.mark.timeout(30)
async def test_recover_telemetry_from_a_short_stream_is_still_warming_up(redis_url: str) -> None:
    stream = stream_name()
    redis = Redis.from_url(redis_url)
    state = FleetState(WINDOW)
    try:
        await append_async(redis, stream, telemetry_record(7, at(0)))
        await append_async(redis, stream, telemetry_record(7, at(600)))
        await recover_telemetry(redis, state, stream=stream)
    finally:
        await redis.delete(stream)
        await redis.aclose()

    assert [p.event_time for p in state.telemetry(7)] == [at(0), at(600)]
    assert state.warming_up(at(600))


@pytest.mark.anyio
@pytest.mark.timeout(30)
async def test_recover_from_empty_streams_resumes_from_the_start(redis_url: str) -> None:
    redis = Redis.from_url(redis_url)
    state = FleetState(WINDOW)
    try:
        telemetry_id = await recover_telemetry(redis, state, stream=stream_name())
        stop_events_id = await recover_stop_events(redis, state, stream=stream_name())
    finally:
        await redis.aclose()

    assert (telemetry_id, stop_events_id) == ("0", "0")
    assert state.clock is None


@pytest.mark.anyio
@pytest.mark.timeout(30)
async def test_recover_stop_events_reads_the_whole_stream_and_skips_malformed(
    redis_url: str,
) -> None:
    stream = stream_name()
    redis = Redis.from_url(redis_url)
    state = FleetState(WINDOW)
    events = [stop_event(7, n, at(n * 60), at(n * 60 + 30)) for n in range(5)]
    try:
        for n, event in enumerate(events):
            last = await append_async(redis, stream, event)
            if n == 2:
                await redis.xadd(stream, {"data": b"not json"})
        resume_id = await recover_stop_events(redis, state, stream=stream, page=2)
    finally:
        await redis.delete(stream)
        await redis.aclose()

    assert state.stop_events(7) == events
    assert resume_id == last


@pytest.mark.anyio
@pytest.mark.timeout(30)
async def test_live_feed_continues_after_the_recovered_entries(redis_url: str) -> None:
    stream = stream_name()
    redis = Redis.from_url(redis_url)
    state = FleetState(WINDOW)
    try:
        await append_async(redis, stream, telemetry_record(7, at(0)))
        await append_async(redis, stream, telemetry_record(7, at(60)))
        resume_id = await recover_telemetry(redis, state, stream=stream)
        feeder = asyncio.create_task(
            feed_telemetry(redis, state, resume_id, stream=stream, block_ms=100)
        )
        try:
            await append_async(redis, stream, telemetry_record(7, at(120)))
            await wait_until(lambda: state.clock == at(120))
        finally:
            await stop(feeder)
    finally:
        await redis.delete(stream)
        await redis.aclose()

    assert [p.event_time for p in state.telemetry(7)] == [at(0), at(60), at(120)]


@pytest.mark.anyio
@pytest.mark.timeout(30)
async def test_state_rebuilt_after_a_restart_matches_the_live_state(redis_url: str) -> None:
    telemetry, stops = stream_name(), stream_name()
    redis = Redis.from_url(redis_url)
    live = FleetState(WINDOW)
    rebuilt = FleetState(WINDOW)
    try:
        feeders = [
            asyncio.create_task(feed_telemetry(redis, live, "0", stream=telemetry, block_ms=100)),
            asyncio.create_task(feed_stop_events(redis, live, "0", stream=stops, block_ms=100)),
        ]
        try:
            for minute in range(40):
                await append_async(redis, telemetry, telemetry_record(7, at(minute * 60)))
                await append_async(redis, telemetry, telemetry_record(8, at(minute * 60 + 5)))
            for n in range(3):
                await append_async(redis, stops, stop_event(7, n, at(n * 600), at(n * 600 + 45)))
            await wait_until(
                lambda: live.clock == at(39 * 60 + 5) and len(live.stop_events(7)) == 3
            )
        finally:
            await stop(*feeders)
        await recover_telemetry(redis, rebuilt, stream=telemetry)
        await recover_stop_events(redis, rebuilt, stream=stops)
    finally:
        await redis.delete(telemetry, stops)
        await redis.aclose()

    live.prune()
    rebuilt.prune()
    assert rebuilt.clock == live.clock
    assert rebuilt.vehicles() == live.vehicles() == [7, 8]
    for tr_id in live.vehicles():
        assert rebuilt.telemetry(tr_id) == live.telemetry(tr_id)
    assert rebuilt.stop_events(7) == live.stop_events(7)


@pytest.mark.anyio
@pytest.mark.timeout(30)
async def test_live_feeds_report_each_entry_after_the_state_has_it(redis_url: str) -> None:
    telemetry, stops = stream_name(), stream_name()
    redis = Redis.from_url(redis_url)
    state = FleetState(WINDOW)
    entries: list[str] = []
    arrivals: list[tuple[int, bool]] = []

    async def arrived(event) -> None:
        arrivals.append((event.stop_id, bool(state.stop_events(event.tr_id))))

    feeders = [
        asyncio.create_task(
            feed_telemetry(
                redis, state, "0", stream=telemetry, block_ms=100, on_entry=entries.append
            )
        ),
        asyncio.create_task(
            feed_stop_events(redis, state, "0", stream=stops, block_ms=100, on_event=arrived)
        ),
    ]
    try:
        entry_id = await append_async(redis, telemetry, telemetry_record(7, at(0)))
        await append_async(redis, stops, stop_event(7, 3, at(0), at(30)))
        await wait_until(lambda: entries and arrivals)
    finally:
        await stop(*feeders)
        await redis.delete(telemetry, stops)
        await redis.aclose()

    assert entries == [entry_id]
    assert arrivals == [(3, True)]
