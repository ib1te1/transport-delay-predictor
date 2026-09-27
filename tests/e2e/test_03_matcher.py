"""Matcher: one arrival per visited stop, in Postgres, on the bus and in the api."""

from datetime import timedelta

import synthetic
from conftest import API_URL, MATCHER_URL, Replayed, Stack, expected_stop_events, parse_time


def _expected() -> set[tuple[int, int, float]]:
    return {
        (bus.tr_id, synthetic.stop_id(bus, k), float(bus.delay_s))
        for bus in synthetic.BUSES
        for k in range(synthetic.VISITED_STOPS)
    }


def test_stop_events_in_postgres(stack: Stack, replayed: Replayed) -> None:
    rows = stack.compose.sql(
        "SELECT tr_id, stop_id, delay_s, time_plan AT TIME ZONE 'UTC',"
        " time_fact AT TIME ZONE 'UTC' FROM stop_events"
    )
    assert {(int(r[0]), int(r[1]), float(r[2])) for r in rows} == _expected()
    assert len(rows) == expected_stop_events()
    buses = {b.tr_id: b for b in synthetic.BUSES}
    for tr_id, stop_id, _delay, plan, fact in rows:
        bus, k = buses[int(tr_id)], int(stop_id) % 1000
        assert parse_time(plan + "+00:00") == synthetic.time_plan(k)
        assert parse_time(fact + "+00:00") == synthetic.arrival(bus, k)


def test_stop_events_on_the_stream(stack: Stack, replayed: Replayed) -> None:
    events = stack.compose.stream("stop_events")
    # at-least-once delivery: a repeat is allowed, a missing event is not
    assert {(e["tr_id"], e["stop_id"], e["delay_s"]) for e in events} == _expected()
    assert len(events) >= expected_stop_events()
    buses = {b.tr_id: b for b in synthetic.BUSES}
    for event in events:
        k = event["stop_id"] % 1000
        assert parse_time(event["time_fact"]) == synthetic.arrival(buses[event["tr_id"]], k)


def test_matcher_quality_counters(stack: Stack, replayed: Replayed) -> None:
    quality = stack.json(f"{MATCHER_URL}/quality")
    assert quality["unread"] is False
    assert quality["stop_events"] == expected_stop_events()
    assert quality["outbox_pending"] == 0
    assert quality["planned_visits"] == len(synthetic.BUSES) * synthetic.PLANNED_STOPS
    assert quality["skipped"]["unknown_vehicle"] == 0
    assert quality["skipped"]["malformed"] == 0
    # the idle terminal has no plan
    assert quality["skipped"]["unplanned"] >= 1


def test_vehicle_card_shows_passed_stops_with_delay(stack: Stack, replayed: Replayed) -> None:
    for bus in synthetic.BUSES:
        card = stack.json(f"{API_URL}/api/vehicles/{bus.tr_id}")
        clock = parse_time(card["clock"])
        assert clock == synthetic.last_event_time()
        assert card["vehicle"]["tr_id"] == bus.tr_id
        span = timedelta(seconds=1800)  # api.card_track_sec
        stops = card["stops"]
        assert stops, card
        for stop in stops:
            plan = parse_time(stop["time_plan"])
            assert clock - span <= plan <= clock + span
            k = stop["stop_id"] % 1000
            assert stop["address"] == synthetic.address(bus, k)
            if k < synthetic.VISITED_STOPS:
                assert stop["delay_s"] == bus.delay_s, stop
                assert parse_time(stop["time_fact"]) == synthetic.arrival(bus, k)
            else:
                assert stop["time_fact"] is None and stop["delay_s"] is None, stop
        track = card["track"]
        assert track and all(clock - span <= parse_time(p["event_time"]) <= clock for p in track)
        last = synthetic.fixes(bus)[-1]
        assert (track[-1]["lat"], track[-1]["lon"]) == (last.lat, last.lon)
