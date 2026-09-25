from datetime import datetime
from enum import StrEnum

from contracts._base import Contract, UtcDatetime


def make_sample_id(tr_id: int, t: datetime) -> str:
    """Prediction point id in the organizers' format: ``<tr_id>_<unix seconds of T>``."""
    if t.utcoffset() is None:
        raise ValueError("make_sample_id requires an aware datetime, got a naive one")
    return f"{tr_id}_{int(t.timestamp())}"


class TelemetryPoint(Contract):
    event_time: UtcDatetime
    lat: float | None
    lon: float | None
    location_valid: bool
    speed_kmh: float | None
    heading_deg: float | None


class ScheduledStop(Contract):
    """A planned stop of the vehicle; ``time_fact`` only if passed no later than T."""

    stop_id: int
    time_plan: UtcDatetime
    lat: float
    lon: float
    time_fact: UtcDatetime | None


class PredictRequest(Contract):
    """Everything the model may use for one prediction point, nothing after ``T``."""

    sample_id: str
    tr_id: int
    T: UtcDatetime
    target_stop_id: int
    target_time_begin: UtcDatetime
    cur_dev_s: float | None
    telemetry: list[TelemetryPoint]
    schedule: list[ScheduledStop]


class ReasonCode(StrEnum):
    accumulated_delay = "accumulated_delay"
    slow_approach = "slow_approach"
    long_dwell = "long_dwell"
    low_speed_segment = "low_speed_segment"
    stale_telemetry = "stale_telemetry"
    unknown = "unknown"


class PredictResponse(Contract):
    sample_id: str
    prediction_s: float
    p_late: float | None
    reasons: list[ReasonCode]
    model_version: str
