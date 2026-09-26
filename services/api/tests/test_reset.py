import psycopg
import pytest
from psycopg import sql

from app.reset import KEPT_TABLES, run_reset


def schema_tables(conn: psycopg.Connection) -> list[str]:
    rows = conn.execute(
        "SELECT tablename FROM pg_tables WHERE schemaname = current_schema()"
    ).fetchall()
    return [name for (name,) in rows]


def row_count(conn: psycopg.Connection, table: str) -> int:
    query = sql.SQL("SELECT count(*) FROM {}").format(sql.Identifier(table))
    return conn.execute(query).fetchone()[0]


def fill(conn: psycopg.Connection) -> None:
    conn.execute("INSERT INTO vehicles (unit_id, tr_id) VALUES (700, 7)")
    conn.execute(
        "INSERT INTO stops_plan (stop_id, tr_id, time_plan, lat, lon)"
        " VALUES (10, 7, '2026-01-06 06:36:00+00', 55.8, 37.4)"
    )
    conn.execute(
        "INSERT INTO predictions (sample_id, tr_id, t, target_stop_id, target_time_begin,"
        " prediction_s, reasons, risk_level, degraded, model_version)"
        " VALUES ('s1', 7, now(), 10, now(), 60, '[]', 'green', false, 'm')"
    )
    conn.execute(
        "INSERT INTO alerts (tr_id, target_stop_id, status, opened_at, predicted_delay_s, reasons)"
        " VALUES (7, 10, 'open', now(), 300, '[]')"
    )
    # A table no migration here knows about stands for one a track adds
    # later: reset must clear it without being edited.
    conn.execute("CREATE TABLE reset_probe (id bigint GENERATED ALWAYS AS IDENTITY, note text)")
    conn.execute("INSERT INTO reset_probe (note) VALUES ('run data')")


def test_run_reset_empties_every_table_except_the_kept_ones(db_conn):
    fill(db_conn)
    kept_before = {t: row_count(db_conn, t) for t in schema_tables(db_conn) if t in KEPT_TABLES}

    cleared = run_reset(db_conn)

    for table in schema_tables(db_conn):
        if table in KEPT_TABLES:
            assert row_count(db_conn, table) == kept_before[table], table
        else:
            assert row_count(db_conn, table) == 0, table
            assert table in cleared
    assert db_conn.execute("SELECT 1 FROM vehicles WHERE unit_id = 700").fetchone()
    assert db_conn.execute("SELECT 1 FROM stops_plan WHERE stop_id = 10").fetchone()
    assert row_count(db_conn, "alembic_version") == 1


def test_run_reset_restarts_identity_columns(db_conn):
    fill(db_conn)
    db_conn.execute("INSERT INTO reset_probe (note) VALUES ('second')")

    run_reset(db_conn)
    db_conn.execute("INSERT INTO reset_probe (note) VALUES ('after reset')")

    assert db_conn.execute("SELECT id FROM reset_probe").fetchone()[0] == 1


def test_run_reset_refuses_to_clear_a_table_a_kept_one_references(db_conn):
    db_conn.execute("CREATE TABLE reset_probe_run (id bigint PRIMARY KEY)")
    db_conn.execute("CREATE TABLE reset_probe_ref (run_id bigint REFERENCES reset_probe_run (id))")

    with pytest.raises(psycopg.errors.FeatureNotSupported):
        run_reset(db_conn, keep={*KEPT_TABLES, "reset_probe_ref"})
