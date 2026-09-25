import asyncio
from contextlib import aclosing
from uuid import uuid4

import pytest
from pydantic import BaseModel
from redis import Redis
from redis.asyncio import Redis as AsyncRedis

from common.bus import append, append_async, read_stream


class Sample(BaseModel):
    n: int


@pytest.mark.anyio
@pytest.mark.timeout(30)
async def test_read_stream_yields_appended_models_and_skips_malformed(redis_url: str) -> None:
    stream = f"test-{uuid4().hex}"
    with Redis.from_url(redis_url) as redis:
        first_id = append(redis, stream, Sample(n=1))
        redis.xadd(stream, {"data": b"not json"})
        append(redis, stream, Sample(n=2))
    client = AsyncRedis.from_url(redis_url)
    got: list[tuple[str, int]] = []
    try:
        async with aclosing(read_stream(client, stream, Sample, last_id="0", block_ms=100)) as it:
            async for entry_id, message in it:
                got.append((entry_id, message.n))
                if len(got) == 2:
                    break
    finally:
        await client.delete(stream)
        await client.aclose()

    assert [n for _, n in got] == [1, 2]
    assert got[0][0] == first_id


@pytest.mark.anyio
@pytest.mark.timeout(30)
async def test_read_stream_from_now_skips_history(redis_url: str) -> None:
    stream = f"test-{uuid4().hex}"
    with Redis.from_url(redis_url) as redis:
        append(redis, stream, Sample(n=1))
    client = AsyncRedis.from_url(redis_url)
    try:
        it = read_stream(client, stream, Sample, block_ms=100)
        # Start reading first: "$" is resolved when the reader begins.
        pending = asyncio.ensure_future(it.__anext__())
        await asyncio.sleep(0.3)
        with Redis.from_url(redis_url) as redis:
            append(redis, stream, Sample(n=2))
        _, message = await asyncio.wait_for(pending, 5)
        await it.aclose()
    finally:
        await client.delete(stream)
        await client.aclose()

    assert message.n == 2


@pytest.mark.anyio
@pytest.mark.timeout(30)
async def test_append_async_is_readable_like_append(redis_url: str) -> None:
    stream = f"test-{uuid4().hex}"
    client = AsyncRedis.from_url(redis_url)
    try:
        entry_id = await append_async(client, stream, Sample(n=1))
        async with aclosing(read_stream(client, stream, Sample, last_id="0", block_ms=100)) as it:
            got_id, message = await it.__anext__()
    finally:
        await client.delete(stream)
        await client.aclose()

    assert (got_id, message.n) == (entry_id, 1)
