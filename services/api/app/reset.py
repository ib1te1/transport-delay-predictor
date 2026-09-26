"""Clear the state of a demo run so the next one starts from nothing.

Run as ``python -m app.reset``, through ``scripts/reset-demo``: api and
matcher keep state in memory, so the script stops them first and starts
them again afterwards. Running this under live services leaves them
serving a run the database no longer has.

The reset is one for the whole system: clearing one track's tables
breaks the tracks that read them. So every table in the schema is
cleared except the kept ones — the reference data seed loads and the
migration version. A table any track adds later is cleared without an
edit here; the kept list changes only with a new seed table.

There is no CASCADE. Every cleared table is in one statement, so it
would add nothing but the power to clear a kept table that references a
cleared one; without it, such a schema fails the reset instead.
"""

import logging
from collections.abc import Collection

import psycopg
from psycopg import sql
from redis import Redis

from app.seed import SEED_TABLES
from common.config import ServiceSettings
from common.db import connect

log = logging.getLogger(__name__)

KEPT_TABLES = frozenset({*SEED_TABLES, "alembic_version"})


def run_reset(conn: psycopg.Connection, keep: Collection[str] = KEPT_TABLES) -> list[str]:
    """Truncate every table in the current schema but ``keep``; returns the cleared ones.

    Identity sequences restart, so ids of the next run start over too.
    Does not commit.
    """
    rows = conn.execute(
        "SELECT tablename FROM pg_tables WHERE schemaname = current_schema() ORDER BY tablename"
    ).fetchall()
    tables = [name for (name,) in rows if name not in keep]
    if tables:
        conn.execute(
            sql.SQL("TRUNCATE {} RESTART IDENTITY").format(
                sql.SQL(", ").join(sql.Identifier(t) for t in tables)
            )
        )
    return tables


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    settings = ServiceSettings()
    with connect(settings.database_url) as conn:
        cleared = run_reset(conn)
        conn.commit()
    log.info("reset: cleared %d tables: %s", len(cleared), ", ".join(cleared))
    # Redis holds nothing but our streams, and nothing there survives a
    # reset, so the whole database goes rather than a list of keys.
    with Redis.from_url(settings.redis_url) as redis:
        redis.flushdb()
    log.info("reset: redis database flushed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
