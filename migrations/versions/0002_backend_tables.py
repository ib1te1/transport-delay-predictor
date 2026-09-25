"""Backend tables: reference data loaded by seed, predictions and alerts.

Revision ID: 0002
Revises: 0001
"""

from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE vehicles (
            unit_id bigint PRIMARY KEY,
            tr_id   bigint NOT NULL
        );
        CREATE INDEX vehicles_tr_id ON vehicles (tr_id);

        CREATE TABLE stops_plan (
            stop_id   bigint PRIMARY KEY,
            tr_id     bigint NOT NULL,
            time_plan timestamptz NOT NULL,
            lat       double precision NOT NULL,
            lon       double precision NOT NULL,
            address   text
        );
        CREATE INDEX stops_plan_tr_time ON stops_plan (tr_id, time_plan);

        CREATE TABLE predictions (
            sample_id         text PRIMARY KEY,
            tr_id             bigint NOT NULL,
            t                 timestamptz NOT NULL,
            target_stop_id    bigint NOT NULL,
            target_time_begin timestamptz NOT NULL,
            cur_dev_s         double precision,
            prediction_s      double precision NOT NULL,
            p_late            double precision,
            reasons           jsonb NOT NULL,
            risk_level        text NOT NULL CHECK (risk_level IN ('green', 'yellow', 'red')),
            degraded          boolean NOT NULL,
            degraded_reason   text,
            model_version     text NOT NULL,
            actual_delay_s    double precision,
            abs_error_s       double precision
        );
        CREATE INDEX predictions_target ON predictions (target_stop_id);
        CREATE INDEX predictions_tr_t ON predictions (tr_id, t DESC);

        CREATE TABLE alerts (
            id                   bigserial PRIMARY KEY,
            tr_id                bigint NOT NULL,
            target_stop_id       bigint NOT NULL,
            segment_from_stop_id bigint,
            status               text NOT NULL
                                 CHECK (status IN ('open', 'confirmed', 'cancelled')),
            opened_at            timestamptz NOT NULL,
            closed_at            timestamptz,
            predicted_delay_s    double precision NOT NULL,
            reasons              jsonb NOT NULL,
            actual_delay_s       double precision,
            lead_time_s          double precision
        );
        CREATE UNIQUE INDEX alerts_one_open ON alerts (tr_id, target_stop_id)
            WHERE status = 'open';
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE alerts, predictions, stops_plan, vehicles")
