import asyncio
import time
from uuid import uuid4

import pytest
from redis import Redis
from redis.asyncio import Redis as AsyncRedis
from redis.client import PubSub

from common.bus import BusMessage, main, publish, subscribe
from common.testing import wait_for_subscribers


def fresh_channel() -> str:
    return f"test-{uuid4().hex}"


def next_message(pubsub: PubSub, timeout: float = 5.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        message = pubsub.get_message(ignore_subscribe_messages=True, timeout=0.1)
        if message is not None:
            return message
    raise AssertionError("no message arrived")


def test_publish_without_subscribers_reaches_nobody(redis_url: str) -> None:
    with Redis.from_url(redis_url) as redis:
        assert publish(redis, fresh_channel(), BusMessage(type="ping", data={})) == 0


@pytest.mark.anyio
async def test_subscribe_yields_published_messages(redis_url: str) -> None:
    channel = fresh_channel()
    sent = BusMessage(type="ping", data={"n": 1})

    async with AsyncRedis.from_url(redis_url) as sub_client:
        stream = subscribe(sub_client, channel)
        received = asyncio.create_task(anext(stream))
        with Redis.from_url(redis_url) as pub:
            await asyncio.to_thread(wait_for_subscribers, pub, channel, 1)
            assert publish(pub, channel, sent) == 1

        assert await asyncio.wait_for(received, 5) == sent
        await stream.aclose()


@pytest.mark.anyio
async def test_subscribe_skips_malformed_messages(redis_url: str) -> None:
    channel = fresh_channel()
    sent = BusMessage(type="ping", data={})

    async with AsyncRedis.from_url(redis_url) as sub_client:
        stream = subscribe(sub_client, channel)
        received = asyncio.create_task(anext(stream))
        with Redis.from_url(redis_url) as pub:
            await asyncio.to_thread(wait_for_subscribers, pub, channel, 1)
            pub.publish(channel, "not json")
            pub.publish(channel, '{"type": "ping"}')
            publish(pub, channel, sent)

        assert await asyncio.wait_for(received, 5) == sent
        await stream.aclose()


def test_cli_publishes_one_message(
    redis_url: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("REDIS_URL", redis_url)
    channel = fresh_channel()

    with Redis.from_url(redis_url) as redis, redis.pubsub() as pubsub:
        pubsub.subscribe(channel)
        wait_for_subscribers(redis, channel, 1)

        assert main([channel, '{"type": "ping", "data": {"n": 1}}']) == 0

        received = next_message(pubsub)

    assert BusMessage.model_validate_json(received["data"]) == BusMessage(
        type="ping", data={"n": 1}
    )
    assert "delivered to 1" in capsys.readouterr().out


def test_cli_rejects_invalid_message() -> None:
    with pytest.raises(SystemExit) as caught:
        main(["dashboard", "not json"])

    assert caught.value.code == 2
