"""Row models of the backend's tables; fields equal columns (checked by tests)."""

from typing import Literal

from pydantic import AwareDatetime, BaseModel

RiskLevel = Literal["green", "yellow", "red"]
AlertStatus = Literal["open", "confirmed", "cancelled"]


class Vehicle(BaseModel):
    unit_id: int
    tr_id: int


class PlanStop(BaseModel):
    stop_id: int
    tr_id: int
    time_plan: AwareDatetime
    lat: float
    lon: float
    address: str | None


class PredictionRow(BaseModel):
    sample_id: str
    tr_id: int
    t: AwareDatetime
    target_stop_id: int
    target_time_begin: AwareDatetime
    cur_dev_s: float | None
    prediction_s: float
    p_late: float | None
    reasons: list[str]
    risk_level: RiskLevel
    degraded: bool
    degraded_reason: str | None
    model_version: str
    actual_delay_s: float | None
    abs_error_s: float | None


class AlertRow(BaseModel):
    id: int
    tr_id: int
    target_stop_id: int
    segment_from_stop_id: int | None
    status: AlertStatus
    opened_at: AwareDatetime
    closed_at: AwareDatetime | None
    predicted_delay_s: float
    reasons: list[str]
    actual_delay_s: float | None
    lead_time_s: float | None
