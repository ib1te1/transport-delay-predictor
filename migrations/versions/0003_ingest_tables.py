"""Ingest telemetry history, delivery outbox and source runs.

Revision ID: 0003
Revises: 0002
"""

from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE ingest_runs (
            run_id              uuid PRIMARY KEY,
            mode                text NOT NULL CHECK (mode IN ('replay', 'emulator')),
            period              text NOT NULL CHECK (period IN ('train', 'test', 'validate')),
            status              text NOT NULL
                                CHECK (status IN ('active', 'completed', 'interrupted')),
            started_at          timestamptz NOT NULL DEFAULT now(),
            file_path           text,
            file_sha256         text,
            start_at            timestamptz,
            end_at              timestamptz,
            speedup             double precision,
            last_event_time     timestamptz,
            last_line_number    bigint,
            dataset_anchor      timestamptz,
            timestamp_shift_s   double precision,
            CHECK ((last_event_time IS NULL) = (last_line_number IS NULL))
        );

        CREATE TABLE telemetry (
            id              bigserial PRIMARY KEY,
            run_id          uuid NOT NULL REFERENCES ingest_runs (run_id),
            source_key      text NOT NULL,
            tr_id           bigint,
            unit_id         bigint NOT NULL,
            event_time      timestamptz NOT NULL,
            lat             double precision,
            lon             double precision,
            location_valid  boolean NOT NULL,
            speed_kmh       double precision,
            heading_deg     double precision,
            source          text NOT NULL
                            CHECK (source IN ('replay', 'emulator', 'emulator_replay')),
            created_at      timestamptz NOT NULL DEFAULT now(),
            published_at    timestamptz,
            stream_id       text,
            UNIQUE (run_id, source_key),
            CHECK (lat IS NULL OR lat BETWEEN -90 AND 90),
            CHECK (lon IS NULL OR lon BETWEEN -180 AND 180),
            CHECK (NOT location_valid OR (lat IS NOT NULL AND lon IS NOT NULL))
        );
        CREATE INDEX telemetry_tr_time ON telemetry (tr_id, event_time DESC);
        CREATE INDEX telemetry_pending ON telemetry (id) WHERE published_at IS NULL;
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE telemetry, ingest_runs")
