import pytest
from factories import at, plan_stop, prediction_row, stop_event, telemetry_record

from app.config import ApiConfig
from app.schemas import VehicleView
from app.views import (
    card_stops,
    diff_views,
    freshness,
    prediction_view,
    summarize,
    track,
    vehicle_view,
)

CONFIG = ApiConfig()


@pytest.mark.parametrize(
    ("silence_s", "expected"),
    [(0, "active"), (120, "active"), (121, "stale"), (900, "stale"), (901, "offline")],
)
def test_freshness_bounds_are_inclusive(silence_s: int, expected: str) -> None:
    assert freshness(at(0), at(silence_s), CONFIG) == expected


def test_position_comes_from_the_last_valid_point_even_if_an_invalid_one_follows() -> None:
    points = [
        telemetry_record(7, at(0), lat=55.1, lon=37.1),
        telemetry_record(7, at(10), lat=55.2, lon=37.2),
        telemetry_record(7, at(20), lat=None, lon=None),
        telemetry_record(7, at(30), unit_id=5, lat=56.0, lon=38.0, location_valid=False),
    ]

    view = vehicle_view(7, points, at(30), None, CONFIG)

    assert (view.lat, view.lon, view.speed_kmh, view.heading_deg) == (55.2, 37.2, 20.0, 90.0)
    assert (view.unit_id, view.last_seen, view.freshness) == (5, at(30), "active")


def test_vehicle_without_any_valid_point_has_no_position() -> None:
    points = [telemetry_record(7, at(0), location_valid=False)]

    view = vehicle_view(7, points, at(200), None, CONFIG)

    assert (view.lat, view.lon, view.heading_deg, view.speed_kmh) == (None, None, None, None)
    assert view.freshness == "stale"


def test_prediction_view_carries_the_row_and_the_address() -> None:
    row = prediction_row(reasons=["accumulated_delay"], risk_level="red", prediction_s=400.0)

    view = prediction_view(row, "Lenina 1")

    assert view.target_address == "Lenina 1"
    assert (view.sample_id, view.t, view.target_stop_id) == (row.sample_id, row.t, 2)
    assert (view.prediction_s, view.risk_level) == (400.0, "red")
    assert view.reasons == ["accumulated_delay"]


def view_at(tr_id: int, seconds: int, *, prediction=None) -> VehicleView:
    """The view of a vehicle seen once, at ``seconds``, with the clock at that moment."""
    point = telemetry_record(tr_id, at(seconds))
    return vehicle_view(tr_id, [point], at(seconds), prediction, CONFIG)


def test_diff_lists_only_changed_vehicles_and_removed_ids() -> None:
    published = {7: view_at(7, 0), 8: view_at(8, 0), 9: view_at(9, 0)}
    current = {7: view_at(7, 0), 8: view_at(8, 5), 10: view_at(10, 0)}

    changed, removed = diff_views(published, current)

    assert [v.tr_id for v in changed] == [8, 10]
    assert removed == [9]


def test_diff_of_equal_views_is_empty() -> None:
    views = {7: view_at(7, 0)}

    assert diff_views(views, dict(views)) == ([], [])


def test_summary_counts_risk_levels_and_freshness() -> None:
    red = prediction_view(prediction_row(risk_level="red"), None)
    green = prediction_view(prediction_row(risk_level="green"), None)
    views = [
        view_at(1, 0, prediction=red),
        view_at(2, 0, prediction=green),
        view_at(3, 0),
        vehicle_view(4, [telemetry_record(4, at(0))], at(200), None, CONFIG),
        vehicle_view(5, [telemetry_record(5, at(0))], at(1000), None, CONFIG),
    ]

    summary = summarize(views)

    assert summary.risk.model_dump() == {"green": 1, "yellow": 0, "red": 1, "none": 3}
    assert summary.freshness.model_dump() == {"active": 3, "stale": 1, "offline": 1}


def test_track_keeps_valid_points_inside_the_window_bounds_included() -> None:
    points = [
        telemetry_record(7, at(0)),
        telemetry_record(7, at(100)),
        telemetry_record(7, at(150), location_valid=False),
        telemetry_record(7, at(200)),
        telemetry_record(7, at(201)),
    ]

    result = track(points, since=at(100), until=at(200))

    assert [p.event_time for p in result] == [at(100), at(200)]
    assert (result[0].lat, result[0].lon, result[0].speed_kmh) == (55.75, 37.62, 20.0)


def test_card_stops_cover_the_window_inclusively_and_carry_the_fact() -> None:
    stops = [
        plan_stop(7, 1, at(-1), address="a"),
        plan_stop(7, 2, at(0), address="b"),
        plan_stop(7, 3, at(600)),
        plan_stop(7, 4, at(1200)),
        plan_stop(7, 5, at(1201)),
    ]
    events = [stop_event(7, 2, at(0), at(45))]

    result = card_stops(stops, events, begin=at(0), end=at(1200))

    assert [s.stop_id for s in result] == [2, 3, 4]
    assert (result[0].address, result[0].time_fact, result[0].delay_s) == ("b", at(45), 45.0)
    assert (result[1].time_fact, result[1].delay_s) == (None, None)
