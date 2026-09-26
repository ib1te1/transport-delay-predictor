"""Postgres history, replay cursor and pending telemetry delivery."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from uuid import UUID, uuid4

import psycopg
from psycopg.rows import dict_row

from contracts import TelemetryRecord


@dataclass(frozen=True)
class RunState:
    run_id: UUID
    status: str
    last_event_time: datetime | None
    last_line_number: int | None
    timestamp_shift_s: float | None

    @property
    def cursor(self) -> tuple[datetime, int] | None:
        if self.last_event_time is None or self.last_line_number is None:
            return None
        return self.last_event_time, self.last_line_number


@dataclass(frozen=True)
class PendingTelemetry:
    id: int
    record: TelemetryRecord


def load_vehicles(conn: psycopg.Connection) -> dict[int, int]:
    """Read the seeded terminal-to-vehicle mapping once for a demo period."""
    return dict(conn.execute("SELECT unit_id, tr_id FROM vehicles").fetchall())


def _run(row: dict) -> RunState:
    return RunState(
        run_id=row["run_id"],
        status=row["status"],
        last_event_time=row["last_event_time"],
        last_line_number=row["last_line_number"],
        timestamp_shift_s=row["timestamp_shift_s"],
    )


def start_replay_run(
    conn: psycopg.Connection,
    *,
    period: str,
    file_path: Path,
    file_sha256: str,
    start_at: datetime | None,
    end_at: datetime | None,
    speedup: float,
    new_run: bool = False,
) -> RunState:
    """Resume the latest replay or explicitly start another one.

    The file fingerprint and time/pacing settings cannot change under an
    existing cursor. The caller owns the transaction.
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT * FROM ingest_runs WHERE mode = 'replay'"
            " ORDER BY started_at DESC, run_id DESC LIMIT 1 FOR UPDATE"
        )
        existing = cur.fetchone()
        if existing is not None and not new_run:
            if existing["status"] == "completed":
                raise ValueError("replay already completed; start a new run explicitly")
            expected = (period, str(file_path), file_sha256, start_at, end_at, speedup)
            actual = tuple(
                existing[name]
                for name in ("period", "file_path", "file_sha256", "start_at", "end_at", "speedup")
            )
            if actual != expected:
                raise ValueError("replay file or settings changed; cannot resume this run")
            cur.execute(
                "UPDATE ingest_runs SET status = 'active' WHERE run_id = %s",
                (existing["run_id"],),
            )
            existing["status"] = "active"
            return _run(existing)
        if existing is not None and existing["status"] == "active":
            raise ValueError("another replay run is active")
        if existing is not None:
            cur.execute(
                "SELECT EXISTS (SELECT 1 FROM telemetry"
                " WHERE run_id = %s AND published_at IS NULL)",
                (existing["run_id"],),
            )
            if cur.fetchone()["exists"]:
                raise ValueError("previous replay still has unpublished telemetry")
        run_id = uuid4()
        cur.execute(
            "INSERT INTO ingest_runs"
            " (run_id, mode, period, status, file_path, file_sha256, start_at, end_at, speedup)"
            " VALUES (%s, 'replay', %s, 'active', %s, %s, %s, %s, %s)",
            (run_id, period, str(file_path), file_sha256, start_at, end_at, speedup),
        )
    return RunState(run_id, "active", None, None, None)


def start_emulator_run(
    conn: psycopg.Connection, *, period: str, dataset_anchor: datetime
) -> RunState:
    """Resume the emulator time offset across process restarts."""
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT * FROM ingest_runs WHERE mode = 'emulator'"
            " ORDER BY started_at DESC, run_id DESC LIMIT 1 FOR UPDATE"
        )
        existing = cur.fetchone()
        if existing is not None and existing["status"] != "completed":
            if existing["period"] != period or existing["dataset_anchor"] != dataset_anchor:
                raise ValueError("emulator period or dataset anchor changed during a run")
            cur.execute(
                "UPDATE ingest_runs SET status = 'active' WHERE run_id = %s",
                (existing["run_id"],),
            )
            existing["status"] = "active"
            return _run(existing)
        run_id = uuid4()
        cur.execute(
            "INSERT INTO ingest_runs"
            " (run_id, mode, period, status, dataset_anchor)"
            " VALUES (%s, 'emulator', %s, 'active', %s)",
            (run_id, period, dataset_anchor),
        )
    return RunState(run_id, "active", None, None, None)


def set_emulator_shift(conn: psycopg.Connection, run_id: UUID, shift_s: float) -> float:
    """First valid packet fixes the clock shift; concurrent clients share it."""
    row = conn.execute(
        "UPDATE ingest_runs SET timestamp_shift_s = COALESCE(timestamp_shift_s, %s)"
        " WHERE run_id = %s RETURNING timestamp_shift_s",
        (shift_s, run_id),
    ).fetchone()
    if row is None:
        raise ValueError(f"unknown ingest run {run_id}")
    return row[0]


def save_telemetry(
    conn: psycopg.Connection,
    run_id: UUID,
    source_key: str,
    record: TelemetryRecord,
    *,
    replay_cursor: tuple[datetime, int] | None = None,
) -> int:
    """Store one point and advance its replay cursor in the same transaction."""
    row = conn.execute(
        "INSERT INTO telemetry"
        " (run_id, source_key, tr_id, unit_id, event_time, lat, lon,"
        " location_valid, speed_kmh, heading_deg, source)"
        " VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
        " ON CONFLICT (run_id, source_key) DO NOTHING"
        " RETURNING id",
        (
            run_id,
            source_key,
            record.tr_id,
            record.unit_id,
            record.event_time,
            record.lat,
            record.lon,
            record.location_valid,
            record.speed_kmh,
            record.heading_deg,
            record.source,
        ),
    ).fetchone()
    if row is None:
        row = conn.execute(
            "SELECT id FROM telemetry WHERE run_id = %s AND source_key = %s",
            (run_id, source_key),
        ).fetchone()
    if replay_cursor is not None:
        event_time, line_number = replay_cursor
        conn.execute(
            "UPDATE ingest_runs SET last_event_time = %s, last_line_number = %s"
            " WHERE run_id = %s AND mode = 'replay'",
            (event_time, line_number, run_id),
        )
    return row[0]


def advance_replay_cursor(
    conn: psycopg.Connection, run_id: UUID, event_time: datetime, line_number: int
) -> None:
    """Account for a skipped CSV row without storing telemetry."""
    conn.execute(
        "UPDATE ingest_runs SET last_event_time = %s, last_line_number = %s"
        " WHERE run_id = %s AND mode = 'replay'",
        (event_time, line_number, run_id),
    )


def pending_telemetry(conn: psycopg.Connection, *, limit: int = 500) -> list[PendingTelemetry]:
    """Read undelivered rows in database order, across interrupted runs."""
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT id, tr_id, unit_id, event_time, lat, lon, location_valid,"
            " speed_kmh, heading_deg, source FROM telemetry"
            " WHERE published_at IS NULL ORDER BY id LIMIT %s",
            (limit,),
        )
        return [
            PendingTelemetry(
                row["id"],
                TelemetryRecord.model_validate({k: v for k, v in row.items() if k != "id"}),
            )
            for row in cur.fetchall()
        ]


def mark_published(conn: psycopg.Connection, row_id: int, stream_id: str) -> None:
    """Record a successful Redis append; a retry may append twice."""
    conn.execute(
        "UPDATE telemetry SET published_at = now(), stream_id = %s"
        " WHERE id = %s AND published_at IS NULL",
        (stream_id, row_id),
    )


def pending_count(conn: psycopg.Connection) -> int:
    return conn.execute("SELECT count(*) FROM telemetry WHERE published_at IS NULL").fetchone()[0]


def pending_backlog(conn: psycopg.Connection) -> tuple[int, float | None]:
    """Return the pending count and age of its oldest row in seconds."""
    count, age = conn.execute(
        "SELECT count(*), EXTRACT(EPOCH FROM now() - min(created_at))"
        " FROM telemetry WHERE published_at IS NULL"
    ).fetchone()
    return count, float(age) if age is not None else None


def is_published(conn: psycopg.Connection, row_id: int) -> bool:
    row = conn.execute(
        "SELECT published_at IS NOT NULL FROM telemetry WHERE id = %s", (row_id,)
    ).fetchone()
    return bool(row and row[0])


def set_run_status(conn: psycopg.Connection, run_id: UUID, status: str) -> None:
    if status not in {"active", "completed", "interrupted"}:
        raise ValueError(f"invalid ingest run status: {status}")
    conn.execute("UPDATE ingest_runs SET status = %s WHERE run_id = %s", (status, run_id))
