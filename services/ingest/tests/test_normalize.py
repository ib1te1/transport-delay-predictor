from datetime import UTC, datetime
from zoneinfo import ZoneInfo

from app.normalize import from_csv, from_ndtp, parse_id

ZONE = ZoneInfo("Europe/Moscow")


def test_csv_time_keeps_microseconds_and_uses_dataset_zone() -> None:
    result = from_csv(
        {
            "unit_id": "664030.0",
            "tr_id": "115106",
            "event_time": "2026-01-06 12:30:31.462764",
            "location_valid": "True",
            "lat": "55.7551234",
            "lon": "37.617321",
            "speed": "35.5",
            "heading": "90",
            "receive_time": "2030-01-01 00:00:00",
        },
        {664030: 115106},
        ZONE,
    )
    assert result.reason is None
    assert result.record is not None
    assert result.record.event_time == datetime(2026, 1, 6, 9, 30, 31, 462764, UTC)
    assert result.record.speed_kmh == 35.5
    assert result.record.source == "replay"


def test_unknown_unit_still_advances_dataset_clock_without_vehicle() -> None:
    result = from_csv(
        {"unit_id": "42", "event_time": "2026-01-06 12:00:00", "location_valid": "False"},
        {},
        ZONE,
    )
    assert result.record is not None
    assert result.record.tr_id is None
    assert result.record.event_time == datetime(2026, 1, 6, 9, tzinfo=UTC)
    assert not result.record.location_valid
    assert result.record.lat is result.record.lon is None


def test_conflicting_csv_vehicle_is_skipped() -> None:
    result = from_csv(
        {"unit_id": "42", "tr_id": "9", "event_time": "2026-01-06 12:00:00"},
        {42: 8},
        ZONE,
    )
    assert result.record is None
    assert result.reason == "tr_id_mismatch"


def test_ndtp_sign_bits_and_shift() -> None:
    result = from_ndtp(
        unit_id=42,
        timestamp=1_788_000_000,
        longitude=376173210,
        latitude=557551234,
        extra_dop=(1 << 5) | (1 << 7),
        speed_avg=28,
        course=180,
        timestamp_shift_s=-10,
        vehicles={42: 9},
    )
    assert result.record is not None
    assert result.record.tr_id == 9
    assert result.record.lat == 55.7551234
    assert result.record.lon == -37.617321
    assert result.record.location_valid
    assert result.record.event_time == datetime.fromtimestamp(1_787_999_990, UTC)


def test_invalid_optional_measurements_do_not_drop_the_point() -> None:
    result = from_csv(
        {
            "unit_id": "42",
            "event_time": "2026-01-06 12:00:00",
            "location_valid": "True",
            "lat": "91",
            "lon": "37",
            "speed": "NaN",
            "heading": "361",
        },
        {42: 9},
        ZONE,
    )
    assert result.record is not None
    assert not result.record.location_valid
    assert result.record.lat is result.record.lon is None
    assert result.record.speed_kmh is None
    assert result.record.heading_deg is None


def test_large_integral_id_does_not_lose_precision() -> None:
    assert parse_id("9007199254740993.0") == 9_007_199_254_740_993
