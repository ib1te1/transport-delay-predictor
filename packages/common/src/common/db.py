"""Postgres access without an ORM: pydantic models in, pydantic models out.

Every connection runs its session in UTC, so timestamptz values always
come back as UTC-aware datetimes whatever the server default is.

Everything here is synchronous. API endpoints that use it are declared
with plain ``def``: FastAPI runs those in a thread pool, and the event
loop serving WebSockets is not blocked.
"""

from collections.abc import Collection, Mapping, Sequence
from datetime import datetime
from typing import Any, LiteralString

import psycopg
from psycopg import sql
from psycopg.rows import class_row
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool
from pydantic import BaseModel
from pydantic_core import to_jsonable_python

# A libpq startup option: applied when the session opens, no extra query.
CONNECT_OPTIONS = "-c TimeZone=UTC"

type Params = Sequence[Any] | Mapping[str, Any] | None


def connect(dsn: str) -> psycopg.Connection:
    """Open one connection with a UTC session."""
    return psycopg.connect(dsn, options=CONNECT_OPTIONS)


def make_pool(dsn: str, *, min_size: int = 1, max_size: int = 10) -> ConnectionPool:
    """Build a pool of UTC-session connections.

    Returned closed: open it with ``with make_pool(dsn) as pool:`` or call
    ``pool.open()`` in the service's startup and ``pool.close()`` on exit.
    """
    return ConnectionPool(
        dsn,
        kwargs={"options": CONNECT_OPTIONS},
        min_size=min_size,
        max_size=max_size,
        open=False,
    )


def fetch_models[M: BaseModel](
    conn: psycopg.Connection,
    model: type[M],
    query: LiteralString | sql.Composable,
    params: Params = None,
) -> list[M]:
    """Run a query and build one model per row, validated by pydantic."""
    with conn.cursor(row_factory=class_row(model)) as cur:
        cur.execute(query, params)
        return cur.fetchall()


def insert_models(
    conn: psycopg.Connection,
    table: str,
    models: Sequence[BaseModel],
    *,
    exclude: Collection[str] = (),
    on_conflict: LiteralString | None = None,
) -> None:
    """Insert models as rows; columns are the model's fields.

    A field with no matching column fails the insert at once. ``dict`` and
    ``list`` values are sent as JSONB; Postgres array columns are not
    supported by this helper. ``on_conflict`` is appended verbatim, e.g.
    ``"ON CONFLICT (id) DO NOTHING"``. Does not commit.
    """
    if not models:
        return
    model_type = type(models[0])
    if any(type(m) is not model_type for m in models):
        raise TypeError("insert_models takes models of a single class")
    unknown = set(exclude) - set(model_type.model_fields)
    if unknown:
        raise ValueError(f"exclude names fields {model_type.__name__} lacks: {sorted(unknown)}")
    columns = [name for name in model_type.model_fields if name not in exclude]

    query = sql.SQL("INSERT INTO {table} ({columns}) VALUES ({values})").format(
        table=sql.Identifier(table),
        columns=sql.SQL(", ").join(sql.Identifier(c) for c in columns),
        values=sql.SQL(", ").join([sql.Placeholder()] * len(columns)),
    )
    if on_conflict:
        query = sql.SQL(" ").join([query, sql.SQL(on_conflict)])

    rows = [_row(model, columns) for model in models]
    with conn.cursor() as cur:
        cur.executemany(query, rows)


def _row(model: BaseModel, columns: list[str]) -> list[Any]:
    values = model.model_dump(include=set(columns))
    return [_adapt(model, name, values[name]) for name in columns]


def _adapt(model: BaseModel, name: str, value: Any) -> Any:
    _reject_naive(value, f"{type(model).__name__}.{name}")
    if isinstance(value, (dict, list)):
        return Jsonb(to_jsonable_python(value))
    return value


def _reject_naive(value: Any, path: str) -> None:
    """Raise if ``value`` is, or contains, a naive datetime.

    Walks dicts and lists/tuples so a timestamp buried in a JSONB payload
    is caught before it is serialized, not silently stored offset-less.
    """
    if isinstance(value, datetime) and value.utcoffset() is None:
        raise ValueError(f"{path} is a naive datetime; timestamps must carry a timezone")
    if isinstance(value, dict):
        for key, item in value.items():
            _reject_naive(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_naive(item, f"{path}[{index}]")
