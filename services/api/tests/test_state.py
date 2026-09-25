from datetime import timedelta

from factories import at, stop_event, telemetry_record

from app.state import FleetState

WINDOW = timedelta(seconds=1800)


def times(state: FleetState, tr_id: int) -> list:
    return [p.event_time for p in state.telemetry(tr_id)]


def test_clock_is_the_latest_event_time_including_unmatched_units() -> None:
    state = FleetState(WINDOW)

    state.add_telemetry(telemetry_record(7, at(0)))
    state.add_telemetry(telemetry_record(None, at(50), unit_id=2))
    state.add_telemetry(telemetry_record(7, at(30)))

    assert state.clock == at(50)
    assert state.vehicles() == [7]


def test_out_of_order_points_are_kept_in_time_order() -> None:
    state = FleetState(WINDOW)

    for seconds in (0, 60, 30):
        state.add_telemetry(telemetry_record(7, at(seconds)))

    assert times(state, 7) == [at(0), at(30), at(60)]


def test_prune_drops_points_older_than_the_window_and_emptied_vehicles() -> None:
    state = FleetState(WINDOW)
    state.add_telemetry(telemetry_record(7, at(0)))
    state.add_telemetry(telemetry_record(7, at(1000)))
    state.add_telemetry(telemetry_record(8, at(100)))
    state.add_telemetry(telemetry_record(9, at(2000)))

    state.prune()

    assert times(state, 7) == [at(1000)]
    assert state.vehicles() == [7, 9]


def test_point_exactly_at_the_window_edge_is_kept() -> None:
    state = FleetState(WINDOW)
    state.add_telemetry(telemetry_record(7, at(0)))
    state.add_telemetry(telemetry_record(7, at(1800)))

    state.prune()

    assert times(state, 7) == [at(0), at(1800)]


def test_repeated_stop_event_replaces_the_earlier_one() -> None:
    state = FleetState(WINDOW)
    first = stop_event(7, 1, at(0), at(30))
    other = stop_event(7, 2, at(100), at(90))
    repeat = stop_event(7, 1, at(0), at(40))

    for event in (other, first, repeat):
        state.add_stop_event(event)

    assert state.stop_events(7) == [repeat, other]
    assert state.stop_events(8) == []


def test_warming_up_until_a_full_window_is_observed() -> None:
    state = FleetState(WINDOW)
    assert state.warming_up(at(0))

    state.add_telemetry(telemetry_record(7, at(0)))

    assert state.warming_up(at(1799))
    assert not state.warming_up(at(1800))


def test_pruning_does_not_restart_the_warm_up() -> None:
    state = FleetState(WINDOW)
    state.add_telemetry(telemetry_record(7, at(0)))
    state.add_telemetry(telemetry_record(7, at(3000)))

    state.prune()

    assert not state.warming_up(at(3000))
    assert state.window == WINDOW
