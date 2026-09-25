from factories import at, plan_stop, prediction_row

from app.models import PredictionRow
from app.store import (
    load_latest_predictions,
    load_plan,
    load_vehicle_predictions,
    save_predictions,
)
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


def test_load_latest_predictions_takes_the_newest_per_vehicle_within_the_window(db_conn) -> None:
    # The local database may hold rows from manual runs; this delete is rolled back.
    db_conn.execute("DELETE FROM predictions")
    rows = [
        prediction_row(sample_id="a1", tr_id=7, t=at(0)),
        prediction_row(sample_id="a2", tr_id=7, t=at(60)),
        prediction_row(sample_id="b1", tr_id=8, t=at(-1)),
        prediction_row(sample_id="c1", tr_id=9, t=at(30)),
    ]
    save_predictions(db_conn, rows)

    latest = load_latest_predictions(db_conn, since=at(0))

    assert [r.sample_id for r in latest] == ["a2", "c1"]
    assert latest[0] == rows[1]


def test_load_vehicle_predictions_is_newest_first_from_since_on(db_conn) -> None:
    db_conn.execute("DELETE FROM predictions")
    save_predictions(
        db_conn,
        [
            prediction_row(sample_id="a0", tr_id=7, t=at(-1)),
            prediction_row(sample_id="a1", tr_id=7, t=at(0)),
            prediction_row(sample_id="a2", tr_id=7, t=at(60)),
            prediction_row(sample_id="b1", tr_id=8, t=at(30)),
        ],
    )

    rows = load_vehicle_predictions(db_conn, 7, since=at(0))

    assert [r.sample_id for r in rows] == ["a2", "a1"]
