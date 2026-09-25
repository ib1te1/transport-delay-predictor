"""Builders for the contract and row models the api tests use."""

from datetime import UTC, datetime, timedelta

from app.models import PlanStop, PredictionRow
from contracts import PredictRequest, StopEvent, TelemetryRecord, make_sample_id

T0 = datetime(2026, 1, 6, 8, 0, tzinfo=UTC)


def at(seconds: float) -> datetime:
    """Dataset time ``seconds`` after ``T0``."""
    return T0 + timedelta(seconds=seconds)


def telemetry_record(
    tr_id: int | None,
    t: datetime,
    *,
    unit_id: int = 1,
    lat: float | None = 55.75,
    lon: float | None = 37.62,
    location_valid: bool = True,
) -> TelemetryRecord:
    return TelemetryRecord(
        tr_id=tr_id,
        unit_id=unit_id,
        event_time=t,
        lat=lat,
        lon=lon,
        location_valid=location_valid,
        speed_kmh=20.0,
        heading_deg=90.0,
        source="replay",
    )


def stop_event(tr_id: int, stop_id: int, time_plan: datetime, time_fact: datetime) -> StopEvent:
    return StopEvent(
        tr_id=tr_id,
        stop_id=stop_id,
        time_plan=time_plan,
        time_fact=time_fact,
        delay_s=(time_fact - time_plan).total_seconds(),
    )


def plan_stop(
    tr_id: int, stop_id: int, time_plan: datetime, *, address: str | None = None
) -> PlanStop:
    return PlanStop(
        stop_id=stop_id, tr_id=tr_id, time_plan=time_plan, lat=55.75, lon=37.62, address=address
    )


def predict_request(tr_id: int, t: datetime, *, cur_dev_s: float | None = None) -> PredictRequest:
    return PredictRequest(
        sample_id=make_sample_id(tr_id, t),
        tr_id=tr_id,
        T=t,
        target_stop_id=1,
        target_time_begin=t + timedelta(minutes=12),
        cur_dev_s=cur_dev_s,
        telemetry=[],
        schedule=[],
    )


def prediction_row(**overrides) -> PredictionRow:
    values = {
        "sample_id": "s1",
        "tr_id": 1,
        "t": T0,
        "target_stop_id": 2,
        "target_time_begin": T0 + timedelta(minutes=12),
        "cur_dev_s": 30.0,
        "prediction_s": 45.0,
        "p_late": None,
        "reasons": ["unknown"],
        "risk_level": "green",
        "degraded": False,
        "degraded_reason": None,
        "model_version": "m1",
        "actual_delay_s": None,
        "abs_error_s": None,
    }
    return PredictionRow(**(values | overrides))
