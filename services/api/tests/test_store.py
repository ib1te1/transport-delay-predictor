from factories import at, plan_stop, prediction_row

from app.models import PredictionRow
from app.store import load_plan, save_predictions
from common.db import fetch_models, insert_models


def test_load_plan_groups_stops_by_vehicle_in_time_order(db_conn) -> None:
    # The local database holds the real seed; this delete is rolled back.
    db_conn.execute("DELETE FROM stops_plan")
    insert_models(
        db_conn,
        "stops_plan",
        [plan_stop(8, 3, at(60)), plan_stop(7, 2, at(120)), plan_stop(7, 1, at(60))],
    )

    plan = load_plan(db_conn)

    assert {tr_id: [s.stop_id for s in stops] for tr_id, stops in plan.items()} == {
        7: [1, 2],
        8: [3],
    }


def test_save_predictions_keeps_the_first_row_for_a_sample(db_conn) -> None:
    first = prediction_row(sample_id="store-test-1", prediction_s=10.0)

    save_predictions(db_conn, [first])
    save_predictions(db_conn, [first.model_copy(update={"prediction_s": 99.0})])

    stored = fetch_models(
        db_conn, PredictionRow, "SELECT * FROM predictions WHERE sample_id = %s", ("store-test-1",)
    )
    assert stored == [first]


def test_save_predictions_with_no_rows_is_a_no_op(db_conn) -> None:
    save_predictions(db_conn, [])
