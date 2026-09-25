"""Typed views of the api and seed sections of config/system.yaml."""

from typing import Literal

from common.config import StrictModel


class RequestConfig(StrictModel):
    telemetry_window_sec: int = 1800
    schedule_back_sec: int = 3600


class RiskConfig(StrictModel):
    green: tuple[float, float] = (-60.0, 120.0)
    red_above: float = 300.0


class ApiConfig(StrictModel):
    scoring_period_sec: int = 60
    stale_after_sec: int = 120
    drop_after_sec: int = 900
    predict_timeout_ms: int = 1000
    card_track_sec: int = 1800
    request: RequestConfig = RequestConfig()
    risk: RiskConfig = RiskConfig()


class SeedConfig(StrictModel):
    period: Literal["train", "test", "validate"] = "test"
    source_timezone: str = "UTC"
