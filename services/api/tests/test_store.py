from factories import at, plan_stop, prediction_row, stop_event

from app.models import PredictionRow
from app.store import (
    Accuracy,
    AlertCounts,
    check_predictions,
    confirm_alerts,
    count_alerts,
    load_accuracy,
    load_alerts,
    load_latest_predictions,
    load_network_stops,
    load_plan,
    load_vehicle_predictions,
    save_predictions,
    update_alerts,
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


def test_load_network_stops_numbers_each_run_by_time_then_id(db_conn) -> None:
    db_conn.execute("DELETE FROM stops_plan")
    insert_models(
        db_conn,
        "stops_plan",
        [
            plan_stop(8, 4, at(60)),
            plan_stop(7, 3, at(120)),
            plan_stop(7, 2, at(60)),
            plan_stop(7, 1, at(60)),
        ],
    )

    stops = load_network_stops(db_conn)

    assert [(s.route_id, s.stop_order, s.stop_id) for s in stops] == [
        (7, 1, 1), (7, 2, 2), (7, 3, 3), (8, 1, 4)
    ]


def test_save_predictions_keeps_the_first_row_for_a_sample(db_conn) -> None:
    first = prediction_row(sample_id="store-test-1", prediction_s=10.0)

    save_predictions(db_conn, [first])
    save_predictions(db_conn, [first.model_copy(update={"prediction_s": 99.0})])

    stored = fetch_models(
        db_conn, PredictionRow, "SELECT * FROM predictions WHERE sample_id = %s", ("store-test-1",)
    )
    assert stored == [first]


def test_save_predictions_returns_only_the_rows_actually_inserted(db_conn) -> None:
    stored = prediction_row(sample_id="store-test-2", prediction_s=10.0)
    save_predictions(db_conn, [stored])

    repeat = stored.model_copy(update={"prediction_s": 99.0})
    fresh = prediction_row(sample_id="store-test-3", prediction_s=20.0)

    saved = save_predictions(db_conn, [repeat, fresh])

    assert saved == [fresh]


def test_save_predictions_returns_only_the_first_row_of_a_repeated_sample_id(db_conn) -> None:
    first = prediction_row(sample_id="store-test-4", prediction_s=10.0)
    second = first.model_copy(update={"prediction_s": 99.0})

    saved = save_predictions(db_conn, [first, second])

    assert saved == [first]
    stored = fetch_models(
        db_conn, PredictionRow, "SELECT * FROM predictions WHERE sample_id = %s", ("store-test-4",)
    )
    assert stored == [first]


def test_save_predictions_with_no_rows_is_a_no_op(db_conn) -> None:
    assert save_predictions(db_conn, []) == []


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


def scored(
    sample_id: str,
    seconds: float,
    *,
    target: int = 20,
    prediction_s: float = 400.0,
    risk: str = "red",
    reasons: tuple[str, ...] = ("accumulated_delay",),
) -> PredictionRow:
    """A prediction of vehicle 7 made at ``at(seconds)`` for stop ``target``."""
    return prediction_row(
        sample_id=sample_id,
        tr_id=7,
        t=at(seconds),
        target_stop_id=target,
        prediction_s=prediction_s,
        risk_level=risk,
        reasons=list(reasons),
    )


def test_an_arrival_checks_every_prediction_made_for_its_stop(db_conn) -> None:
    db_conn.execute("DELETE FROM predictions")
    save_predictions(
        db_conn,
        [
            scored("a", 0, prediction_s=300.0),
            scored("b", 60, prediction_s=450.0),
            scored("c", 120, target=21, prediction_s=10.0),
        ],
    )
    # 400 s late at stop 20
    arrival = stop_event(7, 20, at(720), at(1120))

    changed = check_predictions(db_conn, [arrival])

    rows = fetch_models(db_conn, PredictionRow, "SELECT * FROM predictions ORDER BY sample_id")
    assert changed == 2
    assert [(r.actual_delay_s, r.abs_error_s) for r in rows] == [
        (400.0, 100.0),
        (400.0, 50.0),
        (None, None),
    ]
    assert load_accuracy(db_conn) == Accuracy(75.0, 2)
    # the same arrival again, as after a restart
    assert check_predictions(db_conn, [arrival]) == 0


def test_accuracy_before_any_check_is_empty(db_conn) -> None:
    db_conn.execute("DELETE FROM predictions")
    save_predictions(db_conn, [scored("a", 0)])

    assert load_accuracy(db_conn) == Accuracy(None, 0)
    assert check_predictions(db_conn, []) == 0


def test_red_opens_one_alert_per_pair_and_a_later_red_updates_it(db_conn) -> None:
    db_conn.execute("DELETE FROM alerts")

    [opened] = update_alerts(db_conn, [scored("a", 0)], {"a": 19})
    [updated] = update_alerts(
        db_conn, [scored("b", 60, prediction_s=500.0, reasons=("slow_approach",))], {"b": 99}
    )

    assert (opened.status, opened.opened_at, opened.segment_from_stop_id) == ("open", at(0), 19)
    assert (opened.predicted_delay_s, opened.reasons) == (400.0, ["accumulated_delay"])
    assert updated.id == opened.id
    assert (updated.opened_at, updated.segment_from_stop_id) == (at(0), 19)
    assert (updated.predicted_delay_s, updated.reasons) == (500.0, ["slow_approach"])
    assert len(load_alerts(db_conn, "open")) == 1


def test_below_red_cancels_the_open_alert_and_red_again_opens_a_new_one(db_conn) -> None:
    db_conn.execute("DELETE FROM alerts")
    [opened] = update_alerts(db_conn, [scored("a", 0)], {})

    # another target of the same vehicle leaves the pair alone
    assert update_alerts(db_conn, [scored("b", 30, target=21, risk="yellow")], {}) == []
    [cancelled] = update_alerts(db_conn, [scored("c", 60, prediction_s=200.0, risk="yellow")], {})
    assert update_alerts(db_conn, [scored("d", 120, prediction_s=50.0, risk="green")], {}) == []
    [reopened] = update_alerts(db_conn, [scored("e", 180)], {})

    assert (cancelled.id, cancelled.status, cancelled.closed_at) == (opened.id, "cancelled", at(60))
    assert reopened.id != opened.id
    assert (reopened.status, reopened.opened_at) == ("open", at(180))


def test_an_arrival_confirms_the_open_alert_with_its_lead_time(db_conn) -> None:
    db_conn.execute("DELETE FROM alerts")
    [opened] = update_alerts(db_conn, [scored("a", 0)], {})
    update_alerts(db_conn, [scored("b", 0, target=21)], {})
    update_alerts(db_conn, [scored("c", 60, target=21, risk="yellow")], {})
    arrival = stop_event(7, 20, at(720), at(1120))

    [confirmed] = confirm_alerts(
        db_conn, [arrival, stop_event(7, 21, at(780), at(800)), stop_event(8, 20, at(0), at(0))]
    )

    assert confirmed.id == opened.id
    assert (confirmed.status, confirmed.closed_at) == ("confirmed", at(1120))
    assert (confirmed.actual_delay_s, confirmed.lead_time_s) == (400.0, 1120.0)
    assert confirm_alerts(db_conn, [arrival]) == []
    assert [a.status for a in load_alerts(db_conn, limit=None)] == ["cancelled", "confirmed"]


def test_alerts_are_listed_newest_first_and_counted_by_status(db_conn) -> None:
    db_conn.execute("DELETE FROM alerts")
    update_alerts(db_conn, [scored("a", 0, target=20)], {})
    update_alerts(db_conn, [scored("b", 60, target=21)], {})
    update_alerts(db_conn, [scored("c", 120, target=22)], {})
    update_alerts(db_conn, [scored("d", 180, target=22, risk="yellow")], {})
    confirm_alerts(db_conn, [stop_event(7, 20, at(720), at(900))])

    assert [a.target_stop_id for a in load_alerts(db_conn)] == [22, 21, 20]
    assert [a.target_stop_id for a in load_alerts(db_conn, "open")] == [21]
    assert [a.target_stop_id for a in load_alerts(db_conn, limit=2)] == [22, 21]
    assert count_alerts(db_conn) == AlertCounts(
        open=1, confirmed=1, cancelled=1, mean_lead_time_s=900.0
    )
