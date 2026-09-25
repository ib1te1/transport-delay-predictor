"""Alembic runs SQL written by hand here; there is no autogenerate.

There are no SQLAlchemy table definitions to compare against, and
autogenerate does not see views anyway, so target_metadata stays None.
Each migration runs in its own transaction: DDL in Postgres is
transactional, and a failed migration rolls back whole.
"""

import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import create_engine, pool

# Applies the [loggers]/[handlers] sections of alembic.ini so migration
# progress (e.g. "Running upgrade -> 0001") reaches stderr; without this
# call those sections are inert and alembic's own logger stays silent.
if context.config.config_file_name is not None:
    fileConfig(context.config.config_file_name)


def sqlalchemy_url(database_url: str) -> str:
    """Map the services' postgresql:// URL onto SQLAlchemy's psycopg 3 driver."""
    scheme, sep, rest = database_url.partition("://")
    if not sep or scheme not in ("postgresql", "postgres"):
        raise ValueError(f"DATABASE_URL must start with postgresql://, got {scheme!r}")
    return f"postgresql+psycopg://{rest}"


def run_migrations() -> None:
    if context.is_offline_mode():
        raise RuntimeError("offline mode is not supported: run against a database")
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is not set")
    engine = create_engine(sqlalchemy_url(database_url), poolclass=pool.NullPool)
    with engine.connect() as connection:
        context.configure(
            connection=connection, target_metadata=None, transaction_per_migration=True
        )
        with context.begin_transaction():
            context.run_migrations()


run_migrations()
