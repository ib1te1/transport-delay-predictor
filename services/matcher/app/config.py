"""Matcher runtime settings and dataset-dependent detection thresholds."""

from pathlib import Path

from pydantic import Field

from app.matching import Settings
from common.config import StrictModel, load_section


class MatcherConfig(StrictModel):
    poll_interval_sec: float = Field(default=0.25, gt=0, le=60)
    batch_size: int = Field(default=500, ge=1, le=10_000)
    block_ms: int = Field(default=1000, ge=1, le=60_000)


class MatcherAssumptions(StrictModel):
    stop_radius_m: float = Field(gt=0)
    stopped_speed_kmh: float = Field(gt=0)
    visit_time_tolerance_sec: float = Field(gt=0)
    stale_after_sec: float = Field(gt=0)

    def detector_settings(self) -> Settings:
        return Settings(**self.model_dump())


def load_matcher_settings(path: Path) -> Settings:
    return load_section(path, "matcher", MatcherAssumptions).detector_settings()
