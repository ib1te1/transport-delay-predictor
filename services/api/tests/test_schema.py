import pytest

from app.models import AlertRow, PlanStop, PredictionRow, Vehicle
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


def test_only_one_open_alert_per_vehicle_and_stop(db_conn) -> None:
    insert = (
        "INSERT INTO alerts (tr_id, target_stop_id, status, opened_at, predicted_delay_s, reasons)"
        " VALUES (1, 2, %s, now(), 400, '[]')"
    )
    db_conn.execute(insert, ("confirmed",))
    db_conn.execute(insert, ("open",))
    with pytest.raises(Exception, match="duplicate key"):
        db_conn.execute(insert, ("open",))
