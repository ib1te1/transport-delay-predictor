from datetime import datetime
from uuid import uuid4

import psycopg
import pytest
from pydantic import BaseModel

from common.db import connect
from common.testing import assert_model_matches_table

INNER = """
def test_database(database_url):
    assert database_url


def test_redis(redis_url):
    assert redis_url
"""


@pytest.fixture
def no_infra(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("DATABASE_URL", "REDIS_URL", "REQUIRE_INFRA"):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.usefixtures("no_infra")
def test_missing_variables_skip_locally(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(INNER)

    pytester.runpytest().assert_outcomes(skipped=2)


@pytest.mark.usefixtures("no_infra")
def test_missing_variables_fail_under_require_infra(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("REQUIRE_INFRA", "1")
    pytester.makepyfile(INNER)

    result = pytester.runpytest()

    result.assert_outcomes(errors=2)
    result.stdout.fnmatch_lines(
        [
            "*DATABASE_URL is not set*REQUIRE_INFRA=1*",
            "*REDIS_URL is not set*REQUIRE_INFRA=1*",
        ]
    )


@pytest.mark.usefixtures("no_infra")
def test_set_variables_are_passed_through(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql://example")
    monkeypatch.setenv("REDIS_URL", "redis://example")
    pytester.makepyfile(INNER)

    pytester.runpytest().assert_outcomes(passed=2)


class Stop(BaseModel):
    stop_id: str
    seen_at: datetime


STOP_DDL = """
CREATE TABLE stop (
    stop_id text PRIMARY KEY,
    seen_at timestamptz NOT NULL,
    write_seq bigint GENERATED ALWAYS AS IDENTITY
)
"""


def test_db_conn_rolls_back(pytester: pytest.Pytester, database_url: str) -> None:
    table = f"rollback_probe_{uuid4().hex}"
    pytester.makepyfile(
        f"""
def test_writes(db_conn):
    db_conn.execute("CREATE TABLE {table} (id integer)")
"""
    )

    pytester.runpytest().assert_outcomes(passed=1)

    with connect(database_url) as conn:
        assert conn.execute("SELECT to_regclass(%s)", (table,)).fetchone() == (None,)


def test_model_matches_table_with_allowed_extra_column(db_conn: psycopg.Connection) -> None:
    db_conn.execute(STOP_DDL)

    assert_model_matches_table(db_conn, Stop, "stop", allow={"write_seq"})


def test_column_without_field_is_reported(db_conn: psycopg.Connection) -> None:
    db_conn.execute(STOP_DDL)

    with pytest.raises(AssertionError, match="write_seq"):
        assert_model_matches_table(db_conn, Stop, "stop")


def test_field_without_column_is_reported(db_conn: psycopg.Connection) -> None:
    db_conn.execute(STOP_DDL)

    class StopWithRoute(Stop):
        route_id: str

    with pytest.raises(AssertionError, match="route_id"):
        assert_model_matches_table(db_conn, StopWithRoute, "stop", allow={"write_seq"})


def test_missing_table_is_reported(db_conn: psycopg.Connection) -> None:
    with pytest.raises(AssertionError, match="not found"):
        assert_model_matches_table(db_conn, Stop, "no_such_table")
