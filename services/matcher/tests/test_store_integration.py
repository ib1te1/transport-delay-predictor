"""Stream, recovery and contract checks against the migrated PostgreSQL and Redis."""

import logging
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
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
from app.store import START_ID, StateMismatch, Store
from common.bus import append
from common.config import ServiceSettings
from contracts import StopEvent, TelemetryRecord

NOW = datetime(2026, 1, 6, 12, tzinfo=UTC)
SETTINGS = Settings(50, 5, 300, 300)
CONFIG_DIR = Path(__file__).resolve().parents[3] / "config"
TABLES = (
    "vehicles",
    "stops_plan",
    "matcher_cursor",
    "matcher_state",
    "stop_events",
    "matcher_outbox",
)


@pytest.fixture
def redis(redis_url):
    with Redis.from_url(redis_url) as client:
        yield client


@pytest.fixture
def streams(redis, monkeypatch):
    names = SimpleNamespace(
        telemetry="matcher-test-telemetry-" + uuid4().hex,
        events="matcher-test-events-" + uuid4().hex,
    )
    monkeypatch.setattr("app.store.TELEMETRY_STREAM", names.telemetry)
    monkeypatch.setattr("app.store.STOP_EVENTS_STREAM", names.events)
    yield names
    redis.delete(names.telemetry, names.events)


@pytest.fixture
def pool(database_url):
    schema = "matcher_test_" + uuid4().hex
    with psycopg.connect(database_url, autocommit=True) as admin:
        admin.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
        try:
            for table in TABLES:
                admin.execute(
                    sql.SQL("CREATE TABLE {}.{} (LIKE public.{} INCLUDING ALL)").format(
                        sql.Identifier(schema), sql.Identifier(table), sql.Identifier(table)
                    )
                )
            with ConnectionPool(
                database_url,
                min_size=1,
                max_size=4,
                kwargs={"options": f"-c search_path={schema} -c TimeZone=UTC"},
            ) as pool:
                with pool.connection() as conn:
                    conn.execute("INSERT INTO vehicles (unit_id, tr_id) VALUES (100, 1)")
                    conn.execute(
                        "INSERT INTO stops_plan (stop_id, tr_id, time_plan, lat, lon) "
                        "VALUES (10, 1, %s, 55, 37), (20, 1, %s, 55, 37.05)",
                        (NOW, NOW + timedelta(minutes=10)),
                    )
                yield pool
        finally:
            admin.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))


def make_store(pool, redis, settings: Settings = SETTINGS, **kwargs) -> Store:
    return Store(pool, redis, settings, block_ms=10, **kwargs)


def send(
    redis,
    stream: str,
    at: datetime,
    *,
    lon: float = 37,
    speed: float = 0,
    tr_id: int | None = 1,
    unit_id: int = 100,
    valid: bool = True,
) -> str:
    record = TelemetryRecord(
        tr_id=tr_id,
        unit_id=unit_id,
        event_time=at,
        lat=55 if valid else None,
        lon=lon if valid else None,
        location_valid=valid,
        speed_kmh=speed,
        heading_deg=0,
        source="replay",
    )
    return append(redis, stream, record)


def drain(store: Store) -> int:
    total = 0
    while read := store.process_batch():
        total += read
    return total


def one(pool, query: str):
    with pool.connection() as conn:
        return conn.execute(query).fetchone()


def all_rows(pool, query: str) -> list:
    with pool.connection() as conn:
        return conn.execute(query).fetchall()


def saved_cursor(pool) -> str | None:
    row = one(pool, "SELECT stream_id FROM matcher_cursor")
    return row[0] if row else None


def counts(pool) -> tuple:
    return one(
        pool,
        "SELECT (SELECT count(*) FROM matcher_cursor), (SELECT count(*) FROM matcher_state), "
        "(SELECT count(*) FROM stop_events), (SELECT count(*) FROM matcher_outbox)",
    )


def test_arrival_restart_departure_and_contract(pool, redis, streams):
    first = send(redis, streams.telemetry, NOW)
    store = make_store(pool, redis)
    assert store.process_batch() == 1
    assert saved_cursor(pool) == first
    assert make_store(pool, redis).process_batch() == 0

    restarted = make_store(pool, redis)
    second = send(redis, streams.telemetry, NOW + timedelta(seconds=20), lon=37.01)
    assert restarted.process_batch() == 1
    assert saved_cursor(pool) == second
    assert all_rows(pool, "SELECT time_fact, available_at, departure FROM stop_events") == [
        (NOW, NOW, NOW + timedelta(seconds=20))
    ]
    payloads = all_rows(pool, "SELECT payload FROM matcher_outbox")
    assert len(payloads) == 1
    assert StopEvent.model_validate(payloads[0][0]).delay_s == 0

    assert store.publish_pending(redis) == 1
    assert store.publish_pending(redis) == 0
    entries = redis.xrange(streams.events)
    assert len(entries) == 1
    assert StopEvent.model_validate_json(entries[0][1][b"data"]).stop_id == 10


def test_failed_batch_commits_nothing_and_is_read_again(pool, redis, streams):
    send(redis, streams.telemetry, NOW)
    last = send(redis, streams.telemetry, NOW + timedelta(seconds=20), lon=37.01)
    store = make_store(pool, redis)

    def broken(conn, stream_id):
        raise RuntimeError("disk full")

    store._save_cursor = broken
    with pytest.raises(RuntimeError):
        store.process_batch()
    assert counts(pool) == (0, 0, 0, 0)

    del store._save_cursor
    assert store.process_batch() == 2
    assert store.rejected_late_ticks == 0
    assert saved_cursor(pool) == last
    assert all_rows(pool, "SELECT time_fact, departure FROM stop_events") == [
        (NOW, NOW + timedelta(seconds=20))
    ]
    assert counts(pool) == (1, 1, 1, 1)


@pytest.mark.parametrize("batch_size", [1, 500])
def test_batch_size_does_not_change_results(pool, redis, streams, batch_size):
    stream = streams.telemetry
    passed = NOW + timedelta(minutes=10)
    send(redis, stream, NOW)
    send(redis, stream, NOW + timedelta(seconds=10))
    send(redis, stream, NOW + timedelta(seconds=10))
    send(redis, stream, NOW + timedelta(seconds=15), valid=False)
    send(redis, stream, NOW + timedelta(seconds=20), lon=37.01, speed=30)
    send(redis, stream, NOW + timedelta(seconds=30), tr_id=None, unit_id=999)
    send(redis, stream, NOW + timedelta(seconds=30), tr_id=2, unit_id=200)
    redis.xadd(stream, {"data": "garbage"})
    send(redis, stream, passed, lon=37.0503, speed=30)
    send(redis, stream, passed + timedelta(seconds=10), lon=37.05, speed=30)
    last = send(redis, stream, passed + timedelta(seconds=20), lon=37.06, speed=30)

    store = make_store(pool, redis, batch_size=batch_size)
    assert drain(store) == 11
    assert saved_cursor(pool) == last
    assert all_rows(
        pool,
        "SELECT stop_id, time_fact, available_at, departure, recovered "
        "FROM stop_events ORDER BY stop_id",
    ) == [
        (10, NOW, NOW, NOW + timedelta(seconds=20), False),
        (
            20,
            passed + timedelta(seconds=10),
            passed + timedelta(seconds=20),
            passed + timedelta(seconds=10),
            True,
        ),
    ]
    payloads = [
        StopEvent.model_validate(payload)
        for (payload,) in all_rows(pool, "SELECT payload FROM matcher_outbox ORDER BY id")
    ]
    assert [(e.stop_id, e.delay_s) for e in payloads] == [(10, 0), (20, 10)]
    assert store.rejected_late_ticks == 1
    assert store.quality()["skipped"] == {
        "malformed": 1,
        "unknown_vehicle": 1,
        "unplanned": 1,
        "invalid_location": 1,
    }


def test_recovered_pass_through_is_published_only_after_confirmation(pool, redis, streams):
    store = make_store(pool, redis)
    send(redis, streams.telemetry, NOW, lon=37.0003, speed=20)
    send(redis, streams.telemetry, NOW + timedelta(seconds=10), speed=20)
    assert drain(store) == 2
    assert one(pool, "SELECT count(*) FROM stop_events")[0] == 0
    send(redis, streams.telemetry, NOW + timedelta(seconds=20), lon=37.01, speed=20)
    assert drain(store) == 1
    assert all_rows(pool, "SELECT time_fact, available_at, recovered FROM stop_events") == [
        (NOW + timedelta(seconds=10), NOW + timedelta(seconds=20), True)
    ]


def test_changed_thresholds_stop_the_start(pool, redis, streams, redis_url, monkeypatch, tmp_path):
    send(redis, streams.telemetry, NOW)
    assert make_store(pool, redis).process_batch() == 1

    changed = Settings(60, 5, 300, 300)
    with pytest.raises(StateMismatch, match="assumptions.yaml") as error:
        make_store(pool, redis, changed).load()
    assert "reset-demo" in str(error.value)
    assert "stop_radius_m: saved 50, configured 60" in str(error.value)

    class PoolProxy:
        def open(self, *, wait):
            pass

        def close(self):
            pass

        def connection(self):
            return pool.connection()

    monkeypatch.setattr("app.main.make_pool", lambda _: PoolProxy())
    (tmp_path / "system.yaml").write_bytes((CONFIG_DIR / "system.yaml").read_bytes())
    (tmp_path / "assumptions.yaml").write_text(
        "matcher:\n  stop_radius_m: 60\n  stopped_speed_kmh: 5\n"
        "  visit_time_tolerance_sec: 300\n  stale_after_sec: 300\n",
        encoding="utf-8",
    )
    runtime = ServiceSettings(
        database_url="unused",
        redis_url=redis_url,
        config_path=tmp_path / "system.yaml",
        _env_file=None,
    )
    with pytest.raises(StateMismatch), TestClient(create_app(runtime)):
        pass


def test_demo_reset_starts_a_new_run(pool, redis, streams):
    def run() -> None:
        send(redis, streams.telemetry, NOW)
        send(redis, streams.telemetry, NOW + timedelta(seconds=20), lon=37.01)
        store = make_store(pool, redis)
        assert drain(store) == 2
        assert store.rejected_late_ticks == 0
        assert store.publish_pending(redis) == 1

    run()
    # Same as the api reset: every table but the seed's reference data.
    with pool.connection() as conn:
        tables = [
            name
            for (name,) in conn.execute(
                "SELECT tablename FROM pg_tables WHERE schemaname = current_schema()"
            ).fetchall()
            if name not in ("vehicles", "stops_plan")
        ]
        conn.execute(
            sql.SQL("TRUNCATE {} RESTART IDENTITY").format(
                sql.SQL(", ").join(sql.Identifier(t) for t in tables)
            )
        )
    redis.delete(streams.telemetry, streams.events)

    fresh = make_store(pool, redis)
    fresh.load()
    assert fresh._last_id == START_ID
    quality = fresh.quality()
    assert (quality["stream_cursor"], quality["stop_events"], quality["lag_ms"]) == (START_ID, 0, 0)

    run()
    assert counts(pool) == (1, 1, 1, 0)
    assert redis.xlen(streams.events) == 1


def test_check_stream_warns_about_a_stream_behind_the_cursor(pool, redis, streams, caplog):
    send(redis, streams.telemetry, NOW)
    assert make_store(pool, redis).process_batch() == 1
    redis.delete(streams.telemetry)
    store = make_store(pool, redis)
    store.load()
    with caplog.at_level(logging.WARNING, logger="app.store"):
        store.check_stream()
    assert "reset-demo" in caplog.text


def test_redis_failure_preserves_event(pool, redis, streams):
    send(redis, streams.telemetry, NOW)
    store = make_store(pool, redis)
    assert store.process_batch() == 1

    class UnavailableRedis:
        def xadd(self, *args, **kwargs):
            raise RedisConnectionError("outage")

    with pytest.raises(RedisConnectionError):
        store.publish_pending(UnavailableRedis())
    assert one(pool, "SELECT count(*) FROM matcher_outbox")[0] == 1


def test_service_workers_publish_event_visible_to_api_reader(
    pool, redis, streams, redis_url, monkeypatch
):
    class PoolProxy:
        def open(self, *, wait):
            pass

        def close(self):
            pass

        def connection(self):
            return pool.connection()

    monkeypatch.setattr("app.main.make_pool", lambda _: PoolProxy())
    entry = send(redis, streams.telemetry, NOW)
    runtime = ServiceSettings(
        database_url="unused",
        redis_url=redis_url,
        config_path=CONFIG_DIR / "system.yaml",
        _env_file=None,
    )
    with TestClient(create_app(runtime)) as client:
        deadline = time.monotonic() + 5
        while redis.xlen(streams.events) == 0 and time.monotonic() < deadline:
            time.sleep(0.02)
        assert client.get("/ready").status_code == 200
        quality = client.get("/quality").json()
        assert (quality["stop_events"], quality["stream_cursor"]) == (1, entry)
        entries = redis.xrange(streams.events)
        assert len(entries) == 1
        assert StopEvent.model_validate_json(entries[0][1][b"data"]).tr_id == 1
        client.app.state.store.failing = True
        assert client.get("/ready").json() == {"status": "input_failing"}
