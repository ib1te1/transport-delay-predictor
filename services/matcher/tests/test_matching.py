from datetime import datetime, timedelta

import pytest

from app.dataset import Tick, Visit
from app.matching import Settings, StopDetector

NOW = datetime(2026, 1, 6, 12)
SETTINGS = Settings(50, 5, 1800, 300)


def visit(visit_id="stop", at=NOW, vehicle="bus"):
    return Visit(vehicle, visit_id, at, 37.0, 55.0)


def tick(seconds=0, lon=37.0, speed=0, vehicle="bus"):
    return Tick(vehicle, NOW + timedelta(seconds=seconds), lon, 55.0, speed)


def test_arrival_departure_and_no_duplicate_visit():
    detector = StopDetector([visit()], SETTINGS)
    events = detector.feed(tick())
    assert len(events) == 1
    assert events[0].departure is None
    assert not detector.feed(tick(10))
    assert not detector.feed(tick(20, lon=37.01))
    assert events[0].departure == NOW + timedelta(seconds=20)
    assert not detector.feed(tick(30))
    assert len(detector.events) == 1


def test_pass_through_available_only_after_exit():
    detector = StopDetector([visit()], SETTINGS)
    assert not detector.feed(tick(0, lon=37.0003, speed=20))
    assert not detector.feed(tick(10, speed=20))
    events = detector.feed(tick(20, lon=37.01, speed=20))
    assert len(events) == 1
    assert events[0].recovered
    assert events[0].arrival == NOW + timedelta(seconds=10)
    assert events[0].available_at == NOW + timedelta(seconds=20)


def test_fast_entry_then_stop_uses_first_slow_tick():
    detector = StopDetector([visit()], SETTINGS)
    assert not detector.feed(tick(speed=20))
    event = detector.feed(tick(10))[0]
    assert event.arrival == NOW + timedelta(seconds=10)
    assert not event.recovered


def test_late_and_invalid_ticks_do_not_create_arrivals():
    detector = StopDetector([visit()], SETTINGS)
    assert not detector.feed(tick(10, lon=None))
    assert not detector.feed(tick(0))
    assert detector.rejected_late_ticks == 1
    assert len(detector.feed(tick(20))) == 1


def test_ambiguous_repeated_location_is_not_assigned():
    detector = StopDetector([visit(), visit("second", NOW + timedelta(minutes=10))], SETTINGS)
    assert not detector.feed(tick())


def test_other_vehicle_and_distant_plan_cannot_match():
    detector = StopDetector([visit(vehicle="other"), visit(at=NOW + timedelta(hours=3))], SETTINGS)
    assert not detector.feed(tick())


def test_midnight_arrival_matches_previous_day_visit():
    planned = datetime(2026, 1, 6, 23, 59)
    detector = StopDetector([visit(at=planned)], SETTINGS)
    event = detector.feed(Tick("bus", planned + timedelta(minutes=2), 37, 55, 0))[0]
    assert event.delay_sec == 120


@pytest.mark.parametrize("speed", [0, 20])
def test_gap_does_not_invent_departure_or_recovered_arrival(speed):
    detector = StopDetector([visit()], SETTINGS)
    detector.feed(tick(speed=speed))
    assert not detector.feed(tick(301, lon=37.01))
    if speed == 0:
        assert detector.events[0].departure is None
    else:
        assert not detector.events
