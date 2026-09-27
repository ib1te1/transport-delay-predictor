"""Matcher stream cursor, checkpoint, stop events and delivery outbox.

No cursor row means nothing is processed yet: the matcher reads the
telemetry stream from its start and inserts the row with the first batch.

Revision ID: 0004
Revises: 0003
"""

from alembic import op

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE matcher_cursor (
            id boolean PRIMARY KEY DEFAULT true CHECK (id),
            stream_id text NOT NULL,
            settings jsonb NOT NULL,
            updated_at timestamptz NOT NULL DEFAULT now()
        );

        CREATE TABLE matcher_state (
            tr_id bigint PRIMARY KEY,
            state jsonb NOT NULL,
            updated_at timestamptz NOT NULL DEFAULT now()
        );

        CREATE TABLE stop_events (
            tr_id bigint NOT NULL,
            stop_id bigint NOT NULL,
            time_plan timestamptz NOT NULL,
            time_fact timestamptz NOT NULL,
            delay_s double precision NOT NULL,
            available_at timestamptz NOT NULL,
            departure timestamptz,
            recovered boolean NOT NULL,
            PRIMARY KEY (tr_id, stop_id, time_plan),
            CHECK (available_at >= time_fact),
            CHECK (departure IS NULL OR departure >= time_fact)
        );
        CREATE INDEX stop_events_tr_time ON stop_events (tr_id, time_fact DESC);

        CREATE TABLE matcher_outbox (
            id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            payload jsonb NOT NULL,
            created_at timestamptz NOT NULL DEFAULT now()
        );
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE matcher_outbox, stop_events, matcher_state, matcher_cursor")
