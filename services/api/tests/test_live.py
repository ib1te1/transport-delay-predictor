from datetime import timedelta

from factories import at, plan_stop, prediction_row, stop_event, telemetry_record

from app.config import ApiConfig
from app.live import CurrentPredictions, LiveState
from app.state import FleetState

CONFIG = ApiConfig()
WINDOW = timedelta(seconds=CONFIG.request.telemetry_window_sec)
TARGET = plan_stop(7, 20, at(720), address="Lenina 1")
PLAN = {7: [plan_stop(7, 10, at(-600)), TARGET]}


def live_with(*points) -> LiveState:
    fleet = FleetState(WINDOW)
    for point in points:
        fleet.add_telemetry(point)
    return LiveState(fleet, PLAN)


def row_at(seconds: int, **overrides):
    values = {"sample_id": f"7_{seconds}", "tr_id": 7, "t": at(seconds), "target_stop_id": 20}
    return prediction_row(**(values | overrides))


def test_newer_prediction_replaces_the_held_one_and_an_older_does_not() -> None:
    current = CurrentPredictions()

    current.put(row_at(60))
    current.put(row_at(120))
    current.put(row_at(0))

    assert current.get(7).sample_id == "7_120"
    assert current.vehicles() == [7]


def test_views_show_the_current_prediction_with_the_target_address() -> None:
    live = live_with(telemetry_record(7, at(0)), telemetry_record(8, at(10)))
    live.current.put(row_at(0, risk_level="red"))

    views = live.views(CONFIG)

    assert list(views) == [7, 8]
    assert views[7].prediction.sample_id == "7_0"
    assert views[7].prediction.target_address == "Lenina 1"
    assert views[7].prediction.risk_level == "red"
    assert views[8].prediction is None


def test_prediction_ends_when_the_target_stop_is_passed() -> None:
    live = live_with(telemetry_record(7, at(0)))
    live.current.put(row_at(0))
    live.fleet.add_stop_event(stop_event(7, 10, at(-600), at(-560)))

    assert live.views(CONFIG)[7].prediction is not None

    live.fleet.add_stop_event(stop_event(7, 20, at(720), at(700)))

    assert live.views(CONFIG)[7].prediction is None
    assert live.current.get(7) is None


def test_prediction_ends_when_the_vehicle_goes_offline_and_does_not_come_back() -> None:
    live = live_with(telemetry_record(7, at(0)))
    live.current.put(row_at(0))

    live.fleet.add_telemetry(telemetry_record(8, at(900)))
    assert live.views(CONFIG)[7].freshness == "stale"
    assert live.current.get(7) is not None

    live.fleet.add_telemetry(telemetry_record(8, at(901)))
    offline = live.views(CONFIG)[7]
    assert (offline.freshness, offline.prediction) == ("offline", None)

    live.fleet.add_telemetry(telemetry_record(7, at(905)))
    assert live.views(CONFIG)[7].prediction is None


def test_vehicle_leaves_the_views_once_its_last_point_is_older_than_the_window() -> None:
    live = live_with(telemetry_record(7, at(0)), telemetry_record(8, at(1800)))
    live.current.put(row_at(0))

    assert list(live.views(CONFIG)) == [7, 8]

    live.fleet.add_telemetry(telemetry_record(8, at(1801)))

    assert list(live.views(CONFIG)) == [8]
    assert live.current.get(7) is None


def test_views_are_empty_before_any_telemetry() -> None:
    assert live_with().views(CONFIG) == {}


def test_address_of_an_unknown_stop_is_none() -> None:
    live = live_with()

    assert live.address(20) == "Lenina 1"
    assert live.address(999) is None
