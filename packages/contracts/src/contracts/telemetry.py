from typing import Literal

from contracts._base import Contract, UtcDatetime


class TelemetryRecord(Contract):
    """One telemetry point as ingest publishes it, whatever the stream source.

    ``event_time`` is always dataset time; when the stream comes through the
    emulator, which stamps packets with wall-clock time, ingest shifts it.
    """

    tr_id: int | None
    unit_id: int
    event_time: UtcDatetime
    lat: float | None
    lon: float | None
    location_valid: bool
    speed_kmh: float | None
    heading_deg: float | None
    source: Literal["replay", "emulator", "emulator_replay"]
