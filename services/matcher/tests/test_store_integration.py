"""Contract and recovery checks against the migrated PostgreSQL and Redis."""

import os
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import psycopg
import pytest
from fastapi.testclient import TestClient
from psycopg import sql
from psycopg_pool import ConnectionPool
from redis import ConnectionError as RedisConnectionError
from redis import Redis

from app.main import create_app
from app.matching import Settings
from app.store import Store
from common.config import ServiceSettings
from contracts import StopEvent

NOW = datetime(2026, 1, 6, 12, tzinfo=UTC)
SETTINGS = Settings(50, 5, 300, 300)


def infrastructure_url(name: str) -> str:
    value = os.environ.get(name)
    if value:
        return value
    if os.environ.get("REQUIRE_INFRA") == "1":
        pytest.fail(f"{name} is required with REQUIRE_INFRA=1")
    pytest.skip(f"{name} not set; run with migrated PostgreSQL and Redis")


@pytest.fixture
def store():
    url = infrastructure_url("DATABASE_URL")
    schema = "matcher_test_" + uuid4().hex
    tables = (
        "telemetry",
        "vehicles",
        "stops_plan",
        "matcher_cursor",
        "matcher_state",
        "stop_events",
        "matcher_outbox",
    )
    with psycopg.connect(url, autocommit=True) as admin:
        admin.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
        try:
            for table in tables:
                admin.execute(
                    sql.SQL("CREATE TABLE {}.{} (LIKE public.{} INCLUDING ALL)").format(
                        sql.Identifier(schema), sql.Identifier(table), sql.Identifier(table)
                    )
                )
            with ConnectionPool(
                url,
                min_size=1,
                max_size=4,
                kwargs={"options": f"-c search_path={schema} -c TimeZone=UTC"},
            ) as pool:
                with pool.connection() as conn:
                    conn.execute("INSERT INTO matcher_cursor (id) VALUES (true)")
                    conn.execute(
                        "INSERT INTO stops_plan (stop_id, tr_id, time_plan, lat, lon) "
                        "VALUES (10, 1, %s, 55, 37)",
                        (NOW,),
                    )
                yield Store(pool, SETTINGS)
        finally:
            admin.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))


def telemetry(store: Store, at: datetime, *, lon: float | None = 37, speed: float = 0) -> None:
    with store.pool.connection() as conn:
        conn.execute(
            "INSERT INTO telemetry "
            "(run_id, source_key, tr_id, unit_id, event_time, lat, lon, "
            "location_valid, speed_kmh, heading_deg, source, published_at) "
            "VALUES (%s, %s, 1, 1, %s, 55, %s, true, %s, 0, 'replay', now())",
            (uuid4(), str(uuid4()), at, lon, speed),
        )


def test_arrival_recovery_idempotence_and_api_contract(store: Store, monkeypatch):
    telemetry(store, NOW)
    assert store.process_next()
    assert not store.process_next()
    restarted = Store(store.pool, SETTINGS)
    telemetry(store, NOW + timedelta(seconds=20), lon=37.01)
    assert restarted.process_next()
    with store.pool.connection() as conn:
        event = conn.execute(
            "SELECT time_fact, available_at, departure FROM stop_events"
        ).fetchone()
        assert event == (NOW, NOW, NOW + timedelta(seconds=20))
        assert conn.execute("SELECT count(*) FROM matcher_outbox").fetchone()[0] == 1
        payload = conn.execute("SELECT payload FROM matcher_outbox").fetchone()[0]
        assert StopEvent.model_validate(payload).delay_s == 0

    url = infrastructure_url("REDIS_URL")
    stream = "matcher-test-" + uuid4().hex
    monkeypatch.setattr("app.store.STOP_EVENTS_STREAM", stream)
    with Redis.from_url(url) as redis:
        try:
            assert store.publish_pending(redis) == 1
            assert store.publish_pending(redis) == 0
            entries = redis.xrevrange(stream, count=1)
            assert redis.xlen(stream) == 1
            assert StopEvent.model_validate_json(entries[0][1][b"data"]).stop_id == 10
        finally:
            redis.delete(stream)


def test_unpublished_input_blocks_later_rows_and_redis_failure_preserves_event(store: Store):
    with store.pool.connection() as conn:
        conn.execute(
            "INSERT INTO telemetry "
            "(run_id, source_key, tr_id, unit_id, event_time, lat, lon, "
            "location_valid, speed_kmh, heading_deg, source) "
            "VALUES (%s, %s, 1, 1, %s, 55, 37, true, 0, 0, 'replay')",
            (uuid4(), str(uuid4()), NOW),
        )
    telemetry(store, NOW + timedelta(seconds=1))
    assert not store.process_next()
    with store.pool.connection() as conn:
        conn.execute("UPDATE telemetry SET published_at = now() WHERE published_at IS NULL")
    assert store.process_next()
    assert store.process_next()
    assert not store.process_next()
    with store.pool.connection() as conn:
        assert conn.execute("SELECT count(*) FROM stop_events").fetchone()[0] == 1

    class UnavailableRedis:
        def xadd(self, *args, **kwargs):
            raise RedisConnectionError("outage")

    with pytest.raises(RedisConnectionError):
        store.publish_pending(UnavailableRedis())
    with store.pool.connection() as conn:
        assert conn.execute("SELECT count(*) FROM matcher_outbox").fetchone()[0] == 1


def test_recovered_pass_through_is_published_only_after_confirmation(store: Store):
    telemetry(store, NOW, lon=37.0003, speed=20)
    assert store.process_next()
    with store.pool.connection() as conn:
        assert conn.execute("SELECT count(*) FROM stop_events").fetchone()[0] == 0
    telemetry(store, NOW + timedelta(seconds=10), speed=20)
    assert store.process_next()
    telemetry(store, NOW + timedelta(seconds=20), lon=37.01, speed=20)
    assert store.process_next()
    with store.pool.connection() as conn:
        fact, available, recovered = conn.execute(
            "SELECT time_fact, available_at, recovered FROM stop_events"
        ).fetchone()
    assert (fact, available, recovered) == (
        NOW + timedelta(seconds=10),
        NOW + timedelta(seconds=20),
        True,
    )


def test_service_workers_publish_event_visible_to_api_reader(store: Store, monkeypatch):
    class PoolProxy:
        def open(self, *, wait):
            pass

        def close(self):
            pass

        def connection(self):
            return store.pool.connection()

    monkeypatch.setattr("app.main.make_pool", lambda _: PoolProxy())
    stream = "matcher-test-" + uuid4().hex
    monkeypatch.setattr("app.store.STOP_EVENTS_STREAM", stream)
    telemetry(store, NOW)
    url = infrastructure_url("REDIS_URL")
    config_path = Path(__file__).resolve().parents[3] / "config" / "system.yaml"
    runtime = ServiceSettings(
        database_url="unused", redis_url=url, config_path=config_path, _env_file=None
    )
    with Redis.from_url(url) as redis:
        try:
            with TestClient(create_app(runtime)) as client:
                deadline = time.monotonic() + 5
                while redis.xlen(stream) == 0 and time.monotonic() < deadline:
                    time.sleep(0.02)
                assert client.get("/ready").status_code == 200
                assert client.get("/quality").json()["stop_events"] == 1
                entries = redis.xrange(stream)
                assert len(entries) == 1
                assert StopEvent.model_validate_json(entries[0][1][b"data"]).tr_id == 1
        finally:
            redis.delete(stream)
