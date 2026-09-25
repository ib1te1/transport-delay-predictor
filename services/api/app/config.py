"""Typed views of the api and seed sections of config/system.yaml."""

from typing import Literal

from pydantic import model_validator

from common.config import StrictModel


class RequestConfig(StrictModel):
    telemetry_window_sec: int = 1800
    schedule_back_sec: int = 3600


class RiskConfig(StrictModel):
    green: tuple[float, float] = (-60.0, 120.0)
    red_above: float = 300.0

    @model_validator(mode="after")
    def _check_thresholds_are_ordered(self) -> "RiskConfig":
        low, high = self.green
        if low > high:
            raise ValueError(f"risk.green must be ordered (low, high), got ({low}, {high})")
        if self.red_above < high:
            raise ValueError(
                f"risk.red_above ({self.red_above}) must be >= risk.green's upper bound ({high})"
            )
        return self


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
