"""Turn either telemetry source into the shared, UTC-aware record."""

import math
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from typing import Literal
from zoneinfo import ZoneInfo

from contracts import TelemetryRecord

type Source = Literal["replay", "emulator", "emulator_replay"]


@dataclass(frozen=True)
class RawTelemetry:
    unit_id: int
    event_time: datetime
    lat: float | None
    lon: float | None
    location_valid: bool
    speed_kmh: float | None
    heading_deg: float | None
    source: Source
    tr_id_hint: int | None = None


@dataclass(frozen=True)
class Normalized:
    record: TelemetryRecord | None
    reason: str | None = None


def parse_id(value: object) -> int | None:
    """Parse an integral dataset id without losing bits through a float."""
    if value is None:
        return None
    try:
        number = Decimal(str(value).strip())
    except InvalidOperation:
        return None
    if not number.is_finite() or number != number.to_integral_value():
        return None
    result = int(number)
    return result if -(2**63) <= result < 2**63 else None


def parse_time(value: str, source_zone: ZoneInfo) -> datetime | None:
    """Read dataset event_time, keeping microseconds and converting to UTC."""
    try:
        parsed = datetime.fromisoformat(value.strip())
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=source_zone)
    return parsed.astimezone(UTC)


def _float(value: object) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _valid(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def normalize(raw: RawTelemetry, vehicles: dict[int, int]) -> Normalized:
    """Validate optional measurements and resolve a terminal to a vehicle."""
    if raw.unit_id < 0 or raw.unit_id >= 2**63:
        return Normalized(None, "unit_id")
    if raw.event_time.tzinfo is None:
        return Normalized(None, "event_time")
    tr_id = vehicles.get(raw.unit_id)
    if raw.tr_id_hint is not None and tr_id is not None and raw.tr_id_hint != tr_id:
        return Normalized(None, "tr_id_mismatch")
    lat, lon = raw.lat, raw.lon
    location_valid = (
        raw.location_valid
        and lat is not None
        and lon is not None
        and math.isfinite(lat)
        and math.isfinite(lon)
        and -90 <= lat <= 90
        and -180 <= lon <= 180
    )
    if not location_valid:
        lat = lon = None
    speed = raw.speed_kmh
    if speed is not None and (not math.isfinite(speed) or speed < 0):
        speed = None
    heading = raw.heading_deg
    if heading is not None and (not math.isfinite(heading) or not 0 <= heading <= 360):
        heading = None
    return Normalized(
        TelemetryRecord(
            tr_id=tr_id,
            unit_id=raw.unit_id,
            event_time=raw.event_time.astimezone(UTC),
            lat=lat,
            lon=lon,
            location_valid=location_valid,
            speed_kmh=speed,
            heading_deg=heading,
            source=raw.source,
        )
    )


def from_csv(row: dict[str, str], vehicles: dict[int, int], source_zone: ZoneInfo) -> Normalized:
    """Normalize a traffic.csv row without reading receive_time or labels."""
    unit_id = parse_id(row.get("unit_id"))
    if unit_id is None:
        return Normalized(None, "unit_id")
    event_time = parse_time(row.get("event_time") or "", source_zone)
    if event_time is None:
        return Normalized(None, "event_time")
    hint_text = (row.get("tr_id") or "").strip()
    hint = parse_id(hint_text) if hint_text else None
    if hint_text and hint is None:
        return Normalized(None, "tr_id")
    return normalize(
        RawTelemetry(
            unit_id=unit_id,
            event_time=event_time,
            lat=_float(row.get("lat")),
            lon=_float(row.get("lon")),
            location_valid=_valid(row.get("location_valid")),
            speed_kmh=_float(row.get("speed")),
            heading_deg=_float(row.get("heading")),
            source="replay",
            tr_id_hint=hint,
        ),
        vehicles,
    )


def from_ndtp(
    *,
    unit_id: int,
    timestamp: int,
    longitude: int,
    latitude: int,
    extra_dop: int,
    speed_avg: int,
    course: int,
    timestamp_shift_s: float,
    vehicles: dict[int, int],
) -> Normalized:
    """Normalize decoded G6CellNav00 fields from an emulator packet."""
    try:
        event_time = datetime.fromtimestamp(timestamp, UTC) + timedelta(seconds=timestamp_shift_s)
    except (OverflowError, OSError, ValueError):
        return Normalized(None, "event_time")
    lon_sign = 1 if extra_dop & (1 << 6) else -1
    lat_sign = 1 if extra_dop & (1 << 5) else -1
    return normalize(
        RawTelemetry(
            unit_id=unit_id,
            event_time=event_time,
            lat=lat_sign * latitude / 1e7,
            lon=lon_sign * longitude / 1e7,
            location_valid=bool(extra_dop & (1 << 7)),
            speed_kmh=float(speed_avg),
            heading_deg=float(course),
            source="emulator",
        ),
        vehicles,
    )
