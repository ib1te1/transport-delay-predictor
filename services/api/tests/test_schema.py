from datetime import UTC, datetime

import pytest

from app.models import AlertRow, PlanStop, PredictionRow, Vehicle
from common.db import fetch_models, insert_models
from common.testing import assert_model_matches_table


@pytest.mark.parametrize(
    ("model", "table"),
    [
        (Vehicle, "vehicles"),
        (PlanStop, "stops_plan"),
        (PredictionRow, "predictions"),
        (AlertRow, "alerts"),
    ],
)
def test_row_models_match_tables(db_conn, model, table) -> None:
    assert_model_matches_table(db_conn, model, table)


def prediction_row(**overrides) -> PredictionRow:
    values = {
        "sample_id": "s1",
        "tr_id": 1,
        "t": datetime(2026, 9, 22, 12, 0, tzinfo=UTC),
        "target_stop_id": 2,
        "target_time_begin": datetime(2026, 9, 22, 12, 5, tzinfo=UTC),
        "cur_dev_s": 30.0,
        "prediction_s": 45.0,
        "p_late": 0.5,
        "reasons": ["unknown"],
        "risk_level": "yellow",
        "degraded": False,
        "degraded_reason": None,
        "model_version": "v1",
        "actual_delay_s": None,
        "abs_error_s": None,
    }
    return PredictionRow(**(values | overrides))


def test_prediction_row_reasons_round_trip(db_conn) -> None:
    rows = [
        prediction_row(sample_id="s1", reasons=["unknown"]),
        prediction_row(sample_id="s2", reasons=[]),
    ]

    insert_models(db_conn, "predictions", rows)

    fetched = fetch_models(db_conn, PredictionRow, "SELECT * FROM predictions ORDER BY sample_id")
    assert fetched == rows


def test_alert_row_reasons_round_trip(db_conn) -> None:
    row = AlertRow(
        id=0,
        tr_id=1,
        target_stop_id=2,
        segment_from_stop_id=None,
        status="open",
        opened_at=datetime(2026, 9, 22, 12, 0, tzinfo=UTC),
        closed_at=None,
        predicted_delay_s=40.0,
        reasons=["unknown"],
        actual_delay_s=None,
        lead_time_s=None,
    )

    insert_models(db_conn, "alerts", [row], exclude=("id",))

    [fetched] = fetch_models(db_conn, AlertRow, "SELECT * FROM alerts")
    assert fetched.id > 0
    assert fetched.reasons == ["unknown"]
    assert fetched.model_copy(update={"id": row.id}) == row


def test_only_one_open_alert_per_vehicle_and_stop(db_conn) -> None:
    insert = (
        "INSERT INTO alerts (tr_id, target_stop_id, status, opened_at, predicted_delay_s, reasons)"
        " VALUES (1, 2, %s, now(), 400, '[]')"
    )
    db_conn.execute(insert, ("confirmed",))
    db_conn.execute(insert, ("open",))
    with pytest.raises(Exception, match="duplicate key"):
        db_conn.execute(insert, ("open",))
