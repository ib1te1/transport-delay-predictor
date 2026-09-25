"""pytest plugin for tests that need Postgres or Redis.

Registered through the pytest11 entry point, so it is active in every
test run where common is installed; services need no conftest line.

A missing DATABASE_URL or REDIS_URL skips the test: locally, the rest of
the suite runs without the infrastructure up. CI sets REQUIRE_INFRA=1,
which turns that skip into a failure — otherwise a typo in a variable
name or a dropped service container would leave every database test
skipped under a green run.

Only the process environment is read, never .env: a local .env must not
turn skips into connection errors when the containers are down.
"""

import os
import time
from collections.abc import Collection, Iterator

import psycopg
import pytest
from pydantic import BaseModel
from redis import Redis

from common.db import connect


def require_env(name: str) -> str:
    """Return the variable, or skip the test (fail under REQUIRE_INFRA=1)."""
    value = os.environ.get(name)
    if value:
        return value
    reason = f"{name} is not set"
    if os.environ.get("REQUIRE_INFRA") == "1":
        pytest.fail(f"{reason}, and REQUIRE_INFRA=1 forbids skipping", pytrace=False)
    pytest.skip(reason)


@pytest.fixture
def database_url() -> str:
    return require_env("DATABASE_URL")


@pytest.fixture
def redis_url() -> str:
    return require_env("REDIS_URL")


@pytest.fixture
def db_conn(database_url: str) -> Iterator[psycopg.Connection]:
    """A UTC-session connection whose work is rolled back after the test.

    DDL is transactional in Postgres, so tables created in a test vanish
    too. A test must not commit: that would leak rows into the next one.
    """
    conn = connect(database_url)
    try:
        yield conn
    finally:
        conn.rollback()
        conn.close()


def assert_model_matches_table(
    conn: psycopg.Connection,
    model: type[BaseModel],
    table: str,
    *,
    allow: Collection[str] = (),
) -> None:
    """Fail if the model's fields and the table's columns differ.

    ``allow`` names columns or fields that may exist on one side only,
    such as an identity column the writer never sets.
    """
    rows = conn.execute(
        "SELECT column_name FROM information_schema.columns"
        " WHERE table_schema = current_schema() AND table_name = %s",
        (table,),
    ).fetchall()
    if not rows:
        raise AssertionError(f"table {table!r} not found in the current schema")
    columns = {name for (name,) in rows}
    fields = set(model.model_fields)
    allowed = set(allow)
    only_in_model = sorted(fields - columns - allowed)
    only_in_table = sorted(columns - fields - allowed)
    if only_in_model or only_in_table:
        raise AssertionError(
            f"{model.__name__} vs {table}: fields without a column {only_in_model}, "
            f"columns without a field {only_in_table}"
        )


def subscriber_count(redis: Redis, channel: str) -> int:
    """How many connections are subscribed to ``channel`` right now."""
    counts = {
        (name.decode() if isinstance(name, bytes) else name): count
        for name, count in redis.pubsub_numsub(channel)
    }
    return counts.get(channel, 0)


def wait_for_subscribers(redis: Redis, channel: str, count: int, timeout: float = 5.0) -> None:
    """Block until ``channel`` has at least ``count`` subscribers.

    Pub/sub keeps nothing: a message published before the subscriber is
    listening is lost, so a test waits for the subscription first.
    """
    deadline = time.monotonic() + timeout
    while subscriber_count(redis, channel) < count:
        if time.monotonic() > deadline:
            raise AssertionError(f"{channel!r} did not reach {count} subscriber(s) in {timeout}s")
        time.sleep(0.02)
