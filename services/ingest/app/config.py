"""Validated settings for the NDTP listener and the CSV replay command."""

from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo

from pydantic import Field, field_validator

from common.config import (
    ConfigError,
    DatasetConfig,
    StrictModel,
    load_dataset_config,
    load_section,
)

type Period = Literal["train", "test", "validate"]


class NdtpConfig(StrictModel):
    host: str = "0.0.0.0"
    port: int = Field(default=9201, ge=1, le=65535)
    read_timeout_sec: float = Field(default=10, gt=0, allow_inf_nan=False)
    max_frame_bytes: int = Field(default=65535, ge=25, le=65550)
    dataset_anchor: datetime | None = None

    @field_validator("host")
    @classmethod
    def _host_is_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("host must not be blank")
        return value


class IngestConfig(StrictModel):
    mode: Literal["replay", "emulator"] = "replay"
    ndtp: NdtpConfig = NdtpConfig()


class ReplayConfig(StrictModel):
    period: Period = "test"
    speedup: float = Field(default=60, gt=0, allow_inf_nan=False)
    start_at: datetime | None = None
    end_at: datetime | None = None
    sort_chunk_rows: int = Field(default=20000, gt=0)


class SeedPeriodConfig(StrictModel):
    """Only the period needed to keep replay aligned with seed."""

    period: Period = "test"


def to_utc(value: datetime, source_zone: ZoneInfo) -> datetime:
    """Interpret a naive dataset config time in the shared source zone."""
    return (value if value.tzinfo is not None else value.replace(tzinfo=source_zone)).astimezone(
        UTC
    )


def load_ingest_config(path: Path | str) -> tuple[IngestConfig, DatasetConfig]:
    """Load listener settings and ensure an emulator has a dataset anchor."""
    ingest = load_section(path, "ingest", IngestConfig)
    dataset = load_dataset_config(path)
    if ingest.mode == "emulator" and ingest.ndtp.dataset_anchor is None:
        raise ConfigError(f"{path}: ingest.ndtp.dataset_anchor is required in emulator mode")
    return ingest, dataset


def load_replay_config(path: Path | str) -> tuple[ReplayConfig, DatasetConfig]:
    """Load replay settings and check the seed period before reading any data.

    Reading ``seed.period`` is the narrow exception to the usual
    one-service-one-section rule: a replay against another period's plan
    would produce plausible but wrong predictions.
    """
    ingest, dataset = load_ingest_config(path)
    if ingest.mode != "replay":
        raise ConfigError(f"{path}: ingest.mode must be replay to run CSV replay")
    replay = load_section(path, "replay", ReplayConfig)
    seed = load_section(path, "seed", SeedPeriodConfig)
    if replay.period != seed.period:
        raise ConfigError(
            f"{path}: replay.period ({replay.period}) must equal seed.period ({seed.period})"
        )
    if replay.start_at is not None and replay.end_at is not None:
        if to_utc(replay.start_at, dataset.zone) > to_utc(replay.end_at, dataset.zone):
            raise ConfigError(f"{path}: replay.start_at must not be after replay.end_at")
    return replay, dataset
