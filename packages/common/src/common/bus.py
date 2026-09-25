"""Redis pub/sub between predictor and api.

A bus, not a store: nothing published is kept, and a subscriber that was
not listening at the moment misses the message. After (re)connecting,
the dashboard fetches state over REST, not from here.

Manual publishing, for testing a subscriber without predictor:

    python -m common.bus dashboard '{"type": "ping", "data": {}}'
"""

import argparse
import logging
import os
from collections.abc import AsyncIterator, Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError
from redis import Redis
from redis.asyncio import Redis as AsyncRedis

log = logging.getLogger(__name__)

# The one channel predictor publishes to and api relays to WebSockets.
DASHBOARD_CHANNEL = "dashboard"


class BusMessage(BaseModel):
    """Envelope for everything on the bus.

    The dashboard decides what to refresh by ``type``; the types and their
    ``data`` are defined together with the contracts.
    """

    model_config = ConfigDict(extra="forbid")

    type: str
    data: dict[str, Any]


def publish(redis: Redis, channel: str, message: BusMessage) -> int:
    """Publish synchronously; returns how many subscribers received it."""
    return redis.publish(channel, message.model_dump_json())


async def subscribe(redis: AsyncRedis, channel: str) -> AsyncIterator[BusMessage]:
    """Yield messages from ``channel`` until closed or the connection fails.

    A message that is not a valid envelope is logged and skipped rather
    than ending the subscription. Connection errors propagate: reconnecting
    is the caller's decision.
    """
    pubsub = redis.pubsub(ignore_subscribe_messages=True)
    try:
        await pubsub.subscribe(channel)
        async for raw in pubsub.listen():
            if raw["type"] != "message":
                continue
            try:
                message = BusMessage.model_validate_json(raw["data"])
            except ValidationError:
                log.warning("dropping malformed message on %s: %.200r", channel, raw["data"])
                continue
            yield message
    finally:
        await pubsub.aclose()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m common.bus",
        description="Publish one message to the bus by hand. Reads REDIS_URL, "
        "defaulting to the compose Redis on localhost.",
    )
    parser.add_argument("channel")
    parser.add_argument("message", help='JSON envelope, e.g. {"type": "ping", "data": {}}')
    args = parser.parse_args(argv)
    try:
        message = BusMessage.model_validate_json(args.message)
    except ValidationError as exc:
        parser.error(str(exc))
    url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    with Redis.from_url(url) as redis:
        receivers = publish(redis, args.channel, message)
    print(f"delivered to {receivers} subscriber(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
