from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest
from redis.asyncio import Redis

from app.outbox import Outbox
from app.store import pending_count, save_telemetry, start_replay_run
from common.db import connect, make_pool
from contracts import TelemetryRecord


class OfflineRedis:
    async def xadd(self, *_args, **_kwargs):
        raise ConnectionError("Redis is unavailable")


@pytest.mark.anyio
async def test_saved_point_reaches_redis_after_a_temporary_outage(
    database_url: str, redis_url: str
) -> None:
    stream = f"test-ingest-{uuid4()}"
    with connect(database_url) as conn:
        run = start_replay_run(
            conn,
            period="test",
            file_path=Path("/data/test/traffic.csv"),
            file_sha256=str(uuid4()),
            start_at=None,
            end_at=None,
            speedup=60,
            new_run=True,
        )
        record = TelemetryRecord(
            tr_id=None,
            unit_id=999,
            event_time=datetime(2026, 1, 6, tzinfo=UTC),
            lat=None,
            lon=None,
            location_valid=False,
            speed_kmh=None,
            heading_deg=None,
            source="replay",
        )
        save_telemetry(conn, run.run_id, "2", record)
    pool = make_pool(database_url)
    redis = Redis.from_url(redis_url)
    try:
        with pool:
            with pytest.raises(ConnectionError):
                await Outbox(pool, OfflineRedis(), stream=stream).flush()
            with pool.connection() as conn:
                assert pending_count(conn) >= 1
            assert await Outbox(pool, redis, stream=stream).flush() >= 1
            entries = await redis.xrange(stream)
            assert len(entries) == 1
            assert TelemetryRecord.model_validate_json(entries[0][1][b"data"]) == record
    finally:
        await redis.delete(stream)
        await redis.aclose()
        with connect(database_url) as conn:
            conn.execute("DELETE FROM telemetry WHERE run_id = %s", (run.run_id,))
            conn.execute("DELETE FROM ingest_runs WHERE run_id = %s", (run.run_id,))
