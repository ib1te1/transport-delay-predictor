from datetime import UTC, datetime
from pathlib import Path

import psycopg
import pytest

from app.store import (
    advance_replay_cursor,
    is_published,
    mark_published,
    pending_count,
    pending_telemetry,
    save_telemetry,
    set_emulator_shift,
    set_run_status,
    start_emulator_run,
    start_replay_run,
)
from contracts import TelemetryRecord

T = datetime(2026, 1, 6, tzinfo=UTC)


def point() -> TelemetryRecord:
    return TelemetryRecord(
        tr_id=2,
        unit_id=1,
        event_time=T,
        lat=None,
        lon=None,
        location_valid=False,
        speed_kmh=None,
        heading_deg=None,
        source="replay",
    )


def test_replay_cursor_and_outbox_survive_a_duplicate_input(db_conn: psycopg.Connection) -> None:
    run = start_replay_run(
        db_conn,
        period="test",
        file_path=Path("/data/test/traffic.csv"),
        file_sha256="abcd",
        start_at=None,
        end_at=None,
        speedup=60,
    )
    first_id = save_telemetry(db_conn, run.run_id, "2", point(), replay_cursor=(T, 2))
    again_id = save_telemetry(db_conn, run.run_id, "2", point(), replay_cursor=(T, 2))
    assert first_id == again_id
    assert pending_count(db_conn) == 1
    assert pending_telemetry(db_conn)[0].record == point()
    assert not is_published(db_conn, first_id)
    mark_published(db_conn, first_id, "1-0")
    assert is_published(db_conn, first_id)
    assert pending_count(db_conn) == 0
    advance_replay_cursor(db_conn, run.run_id, T, 3)
    set_run_status(db_conn, run.run_id, "interrupted")
    resumed = start_replay_run(
        db_conn,
        period="test",
        file_path=Path("/data/test/traffic.csv"),
        file_sha256="abcd",
        start_at=None,
        end_at=None,
        speedup=60,
    )
    assert resumed.run_id == run.run_id
    assert resumed.cursor == (T, 3)


def test_emulator_shift_is_fixed_by_the_first_packet(db_conn: psycopg.Connection) -> None:
    run = start_emulator_run(db_conn, period="test", dataset_anchor=T)
    assert set_emulator_shift(db_conn, run.run_id, -123.5) == -123.5
    assert set_emulator_shift(db_conn, run.run_id, 999.0) == -123.5
    resumed = start_emulator_run(db_conn, period="test", dataset_anchor=T)
    assert resumed.timestamp_shift_s == -123.5


def test_replay_cannot_resume_with_another_file(db_conn: psycopg.Connection) -> None:
    start_replay_run(
        db_conn,
        period="test",
        file_path=Path("/data/test/traffic.csv"),
        file_sha256="abcd",
        start_at=None,
        end_at=None,
        speedup=60,
    )
    with pytest.raises(ValueError, match="file or settings changed"):
        start_replay_run(
            db_conn,
            period="test",
            file_path=Path("/data/test/traffic.csv"),
            file_sha256="different",
            start_at=None,
            end_at=None,
            speedup=60,
        )
