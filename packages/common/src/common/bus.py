"""Redis messaging between the services: Streams for data, pub/sub for fan-out.

Streams (``append``/``read_stream``) carry the data flow between services —
``telemetry``, ``stop_events``, ``predictions`` — and keep their entries up
to the approximate cap, so a reader that restarts resumes instead of losing
what it missed. Pub/sub (``publish``/``subscribe``, ``BusMessage``) is only
the api's fan-out to dashboard WebSockets: nothing published there is kept,
and a subscriber that was not listening at the moment misses the message.
After (re)connecting, the dashboard fetches state over REST, not from here.

Manual publishing, for testing the dashboard relay without api:

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

# The one channel api relays to dashboard WebSockets; predictor publishes
# nothing, it only answers PredictRequest over HTTP.
DASHBOARD_CHANNEL = "dashboard"


class BusMessage(BaseModel):
    """Envelope for pub/sub messages.

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


# Streams carry the data flow between services. Unlike pub/sub, a reader
# that restarts continues from the last id it saw instead of losing what
# was published meanwhile.
TELEMETRY_STREAM = "telemetry"
STOP_EVENTS_STREAM = "stop_events"
PREDICTIONS_STREAM = "predictions"

# Approximate cap per stream; old entries are trimmed. The database, not
# the bus, is the history.
STREAM_MAXLEN = 200_000


def append(redis: Redis, stream: str, message: BaseModel, *, maxlen: int = STREAM_MAXLEN) -> str:
    """Append a model as one stream entry; returns the entry id."""
    entry_id = redis.xadd(
        stream, {"data": message.model_dump_json()}, maxlen=maxlen, approximate=True
    )
    return entry_id.decode() if isinstance(entry_id, bytes) else entry_id


async def read_stream[M: BaseModel](
    redis: AsyncRedis,
    stream: str,
    model: type[M],
    *,
    last_id: str = "$",
    block_ms: int = 5000,
    count: int = 500,
) -> AsyncIterator[tuple[str, M]]:
    """Yield ``(entry_id, model)`` for entries after ``last_id``, forever.

    ``"$"`` means "only what arrives from now on"; pass a saved id to
    resume, or ``"0"`` to read from the start. A malformed entry is logged
    and skipped. Connection errors propagate: reconnecting is the caller's
    decision, and the caller resumes from the last id it received. Entries
    older than the approximate ``STREAM_MAXLEN`` cap are trimmed; a reader
    resuming from a trimmed id continues from the oldest surviving entry.
    """
    if last_id == "$":
        newest = await redis.xrevrange(stream, count=1)
        last_id = _decode(newest[0][0]) if newest else "0"
    while True:
        batches = await redis.xread({stream: last_id}, count=count, block=block_ms)
        for _name, entries in batches or []:
            for raw_id, fields in entries:
                last_id = _decode(raw_id)
                raw = fields.get(b"data", fields.get("data"))
                try:
                    yield last_id, model.model_validate_json(raw)
                except ValidationError:
                    log.warning("dropping malformed entry %s on %s: %.200r", last_id, stream, raw)


def _decode(value: bytes | str) -> str:
    return value.decode() if isinstance(value, bytes) else value


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
