from datetime import UTC, datetime, timedelta, timezone
from typing import Any

import psycopg
import pytest
from psycopg import errors
from pydantic import BaseModel

from common.db import fetch_models, insert_models, make_pool

MSK = timezone(timedelta(hours=3))

READING_DDL = """
CREATE TABLE reading (
    id integer PRIMARY KEY,
    taken_at timestamptz NOT NULL,
    payload jsonb NOT NULL,
    note text,
    created_at timestamptz NOT NULL DEFAULT now()
)
"""
SELECT_READINGS = "SELECT id, taken_at, payload, note FROM reading ORDER BY id"


class Reading(BaseModel):
    id: int
    taken_at: datetime
    payload: dict[str, Any]
    note: str | None = None


class ReadingWithSource(Reading):
    source: str = "sensor"


TAGGED_DDL = """
CREATE TABLE tagged (
    id integer PRIMARY KEY,
    tags jsonb NOT NULL
)
"""
SELECT_TAGGED = "SELECT id, tags FROM tagged ORDER BY id"


class Tagged(BaseModel):
    id: int
    tags: list[str]


@pytest.fixture
def conn(db_conn: psycopg.Connection) -> psycopg.Connection:
    db_conn.execute(READING_DDL)
    return db_conn


def reading(id: int, **overrides: Any) -> Reading:
    values: dict[str, Any] = {
        "id": id,
        "taken_at": datetime(2026, 9, 22, 12, 0, tzinfo=MSK),
        "payload": {"speed": 12.5},
        "note": "ok",
    }
    return Reading(**(values | overrides))


def test_models_round_trip(conn: psycopg.Connection) -> None:
    readings = [reading(1), reading(2, note=None, payload={})]

    insert_models(conn, "reading", readings)

    assert fetch_models(conn, Reading, SELECT_READINGS) == readings


def test_timestamps_come_back_in_utc(conn: psycopg.Connection) -> None:
    insert_models(conn, "reading", [reading(1)])

    [row] = fetch_models(conn, Reading, SELECT_READINGS)

    assert row.taken_at.utcoffset() == timedelta(0)
    assert row.taken_at == datetime(2026, 9, 22, 9, 0, tzinfo=UTC)


def test_session_time_zone_is_utc(database_url: str, db_conn: psycopg.Connection) -> None:
    with make_pool(database_url) as pool, pool.connection() as pooled:
        assert pooled.execute("SHOW TimeZone").fetchone() == ("UTC",)
    assert db_conn.execute("SHOW TimeZone").fetchone() == ("UTC",)


def test_dict_fields_are_stored_as_jsonb(conn: psycopg.Connection) -> None:
    payload = {"stops": [1, 2], "meta": {"at": datetime(2026, 9, 22, tzinfo=UTC)}}

    insert_models(conn, "reading", [reading(1, payload=payload)])

    assert conn.execute("SELECT jsonb_typeof(payload) FROM reading").fetchone() == ("object",)
    [row] = fetch_models(conn, Reading, SELECT_READINGS)
    assert row.payload == {"stops": [1, 2], "meta": {"at": "2026-09-22T00:00:00Z"}}


def test_list_fields_are_stored_as_jsonb(conn: psycopg.Connection) -> None:
    conn.execute(TAGGED_DDL)
    rows = [Tagged(id=1, tags=["a", "b"]), Tagged(id=2, tags=[])]

    insert_models(conn, "tagged", rows)

    assert conn.execute("SELECT jsonb_typeof(tags) FROM tagged ORDER BY id").fetchall() == [
        ("array",),
        ("array",),
    ]
    assert fetch_models(conn, Tagged, SELECT_TAGGED) == rows


def test_naive_datetime_is_rejected(conn: psycopg.Connection) -> None:
    naive = reading(1, taken_at=datetime(2026, 9, 22, 12, 0))

    with pytest.raises(ValueError, match="naive"):
        insert_models(conn, "reading", [naive])


def test_naive_datetime_nested_in_payload_is_rejected(conn: psycopg.Connection) -> None:
    payload = {"meta": {"at": datetime(2026, 9, 22, 12, 0)}}

    with pytest.raises(ValueError, match=r"Reading\.payload\.meta\.at is a naive datetime"):
        insert_models(conn, "reading", [reading(1, payload=payload)])


def test_on_conflict_clause_is_applied(conn: psycopg.Connection) -> None:
    insert_models(conn, "reading", [reading(1, note="first")])

    insert_models(
        conn, "reading", [reading(1, note="second")], on_conflict="ON CONFLICT (id) DO NOTHING"
    )

    assert [r.note for r in fetch_models(conn, Reading, SELECT_READINGS)] == ["first"]


def test_field_without_column_fails(conn: psycopg.Connection) -> None:
    extra = ReadingWithSource(**reading(1).model_dump())

    with pytest.raises(errors.UndefinedColumn):
        insert_models(conn, "reading", [extra])


def test_excluded_field_is_not_written(conn: psycopg.Connection) -> None:
    extra = ReadingWithSource(**reading(1).model_dump())

    insert_models(conn, "reading", [extra], exclude={"source"})

    assert len(fetch_models(conn, Reading, SELECT_READINGS)) == 1


def test_excluding_unknown_field_is_an_error(conn: psycopg.Connection) -> None:
    with pytest.raises(ValueError, match="sourse"):
        insert_models(conn, "reading", [reading(1)], exclude={"sourse"})


def test_mixed_model_classes_are_rejected(conn: psycopg.Connection) -> None:
    mixed = [reading(1), ReadingWithSource(**reading(2).model_dump())]

    with pytest.raises(TypeError):
        insert_models(conn, "reading", mixed)


def test_empty_insert_is_a_no_op(conn: psycopg.Connection) -> None:
    insert_models(conn, "reading", [])

    assert fetch_models(conn, Reading, SELECT_READINGS) == []
