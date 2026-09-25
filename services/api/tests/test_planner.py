from datetime import timedelta

import pytest
from factories import at, plan_stop, stop_event, telemetry_record

from app.config import ApiConfig
from app.planner import build_request, plan_tick, select_target
from app.state import FleetState

CONFIG = ApiConfig()
T = at(3600)
MIN = timedelta(minutes=1)
SEC = timedelta(seconds=1)
PLAN = {7: [plan_stop(7, 1, T + 12 * MIN)], 8: [plan_stop(8, 2, T + 12 * MIN)]}


def fleet(*points: tuple[int, object]) -> FleetState:
    state = FleetState(timedelta(seconds=CONFIG.request.telemetry_window_sec))
    for tr_id, t in points:
        state.add_telemetry(telemetry_record(tr_id, t))
    return state


def test_select_target_excludes_exactly_ten_minutes_and_includes_fifteen() -> None:
    ten = plan_stop(7, 1, T + 10 * MIN)
    fifteen = plan_stop(7, 2, T + 15 * MIN)

    assert select_target([ten, fifteen], T) == fifteen


def test_select_target_takes_the_first_stop_in_the_window() -> None:
    stops = [plan_stop(7, 1, T + 11 * MIN), plan_stop(7, 2, T + 13 * MIN)]

    assert select_target(stops, T).stop_id == 1


def test_select_target_without_a_stop_in_the_window_is_none() -> None:
    stops = [plan_stop(7, 1, T + 5 * MIN), plan_stop(7, 2, T + 15 * MIN + SEC)]

    assert select_target(stops, T) is None


def test_build_request_sees_nothing_after_t() -> None:
    telemetry = [
        telemetry_record(7, T - 60 * SEC),
        telemetry_record(7, T),
        telemetry_record(7, T + SEC),
    ]
    passed = stop_event(7, 1, T - 10 * MIN, T - 9 * MIN)
    future = stop_event(7, 2, T - 2 * MIN, T + 30 * SEC)
    target = plan_stop(7, 3, T + 12 * MIN)
    stops = [plan_stop(7, 1, T - 10 * MIN), plan_stop(7, 2, T - 2 * MIN), target]

    request = build_request(7, T, target, stops, telemetry, [passed, future], CONFIG)

    assert [p.event_time for p in request.telemetry] == [T - 60 * SEC, T]
    assert {s.stop_id: s.time_fact for s in request.schedule} == {
        1: passed.time_fact,
        2: None,
        3: None,
    }
    assert request.cur_dev_s == 60.0


def test_build_request_limits_telemetry_and_schedule_to_their_windows() -> None:
    telemetry = [
        telemetry_record(7, T - 1801 * SEC),
        telemetry_record(7, T - 1800 * SEC),
        telemetry_record(7, T),
    ]
    target = plan_stop(7, 3, T + 12 * MIN)
    stops = [
        plan_stop(7, 1, T - 3601 * SEC),
        plan_stop(7, 2, T - 3600 * SEC),
        target,
        plan_stop(7, 4, T + 14 * MIN),
    ]

    request = build_request(7, T, target, stops, telemetry, [], CONFIG)

    assert [p.event_time for p in request.telemetry] == [T - 1800 * SEC, T]
    assert [s.stop_id for s in request.schedule] == [2, 3]
    assert request.cur_dev_s is None
    assert request.sample_id == f"7_{int(T.timestamp())}"
    assert (request.tr_id, request.T) == (7, T)
    assert (request.target_stop_id, request.target_time_begin) == (3, target.time_plan)


def test_cur_dev_s_is_the_delay_at_the_latest_passed_stop() -> None:
    early = stop_event(7, 1, T - 20 * MIN, T - 19 * MIN)
    late = stop_event(7, 2, T - 10 * MIN, T - 8 * MIN)
    target = plan_stop(7, 3, T + 12 * MIN)

    request = build_request(7, T, target, [target], [telemetry_record(7, T)], [late, early], CONFIG)

    assert request.cur_dev_s == 120.0


def test_plan_tick_skips_vehicles_without_a_target() -> None:
    state = fleet((7, T), (9, T))

    candidates = plan_tick(state, PLAN | {9: []}, T, CONFIG)

    assert [c.request.tr_id for c in candidates] == [7]


@pytest.mark.parametrize(
    ("silence_s", "expected"),
    [(120, "fresh"), (121, "stale"), (900, "stale"), (901, "dropped")],
)
def test_plan_tick_marks_stale_and_drops_silent_vehicles(silence_s: int, expected: str) -> None:
    state = fleet((8, T), (7, T - silence_s * SEC))

    by_vehicle = {c.request.tr_id: c for c in plan_tick(state, PLAN, T, CONFIG)}

    if expected == "dropped":
        assert 7 not in by_vehicle
    else:
        assert by_vehicle[7].stale is (expected == "stale")
    assert by_vehicle[8].stale is False


def test_plan_tick_flags_warm_up_until_a_full_window_is_seen() -> None:
    short = fleet((7, T - 60 * SEC), (7, T))
    full = fleet((7, T - 1800 * SEC), (7, T))

    assert [c.warming_up for c in plan_tick(short, PLAN, T, CONFIG)] == [True]
    assert [c.warming_up for c in plan_tick(full, PLAN, T, CONFIG)] == [False]
