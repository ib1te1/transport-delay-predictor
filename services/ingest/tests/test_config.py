from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
from pydantic import ValidationError

from app.config import (
    IngestConfig,
    NdtpConfig,
    ReplayConfig,
    load_ingest_config,
    load_replay_config,
    to_utc,
)
from common.config import REPO_ROOT, ConfigError

SYSTEM_YAML = REPO_ROOT / "config" / "system.yaml"


def write_config(
    tmp_path: Path, *, mode: str = "replay", seed: str = "test", replay: str = "test"
) -> Path:
    path = tmp_path / "system.yaml"
    path.write_text(
        "dataset:\n"
        "  source_timezone: Europe/Moscow\n"
        "ingest:\n"
        f"  mode: {mode}\n"
        "  ndtp:\n"
        "    dataset_anchor: 2026-01-06 12:00:00\n"
        "seed:\n"
        f"  period: {seed}\n"
        "replay:\n"
        f"  period: {replay}\n",
        encoding="utf-8",
    )
    return path


def test_repository_config_loads() -> None:
    ingest, _dataset = load_ingest_config(SYSTEM_YAML)
    replay, _dataset = load_replay_config(SYSTEM_YAML)
    assert ingest.mode == "replay"
    assert replay.period == "test"


@pytest.mark.parametrize(
    ("kwargs", "field"),
    [
        ({"port": 0}, "port"),
        ({"read_timeout_sec": 0}, "read_timeout_sec"),
        ({"max_frame_bytes": 24}, "max_frame_bytes"),
        ({"host": "  "}, "host"),
    ],
)
def test_ndtp_config_rejects_invalid_values(kwargs: dict, field: str) -> None:
    with pytest.raises(ValidationError, match=field):
        NdtpConfig(**kwargs)


def test_replay_config_rejects_invalid_speedup() -> None:
    with pytest.raises(ValidationError, match="speedup"):
        ReplayConfig(speedup=0)


def test_replay_rejects_another_seed_period(tmp_path: Path) -> None:
    path = write_config(tmp_path, seed="train", replay="test")
    with pytest.raises(ConfigError, match="replay.period .* seed.period"):
        load_replay_config(path)


def test_replay_rejects_emulator_mode(tmp_path: Path) -> None:
    path = write_config(tmp_path, mode="emulator")
    with pytest.raises(ConfigError, match="ingest.mode"):
        load_replay_config(path)


def test_emulator_requires_an_anchor(tmp_path: Path) -> None:
    path = write_config(tmp_path, mode="emulator")
    path.write_text(
        path.read_text(encoding="utf-8").replace("2026-01-06 12:00:00", "null"),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="dataset_anchor"):
        load_ingest_config(path)


def test_replay_compares_naive_and_aware_bounds_in_source_zone(tmp_path: Path) -> None:
    path = write_config(tmp_path)
    with path.open("a", encoding="utf-8") as file:
        file.write("  start_at: 2026-01-06 12:00:00\n  end_at: 2026-01-06T08:59:59+00:00\n")
    with pytest.raises(ConfigError, match="start_at"):
        load_replay_config(path)


def test_naive_dataset_time_uses_the_shared_zone() -> None:
    value = datetime(2026, 1, 6, 12, 0, 0)
    assert to_utc(value, ZoneInfo("Europe/Moscow")) == datetime(2026, 1, 6, 9, 0, 0, tzinfo=UTC)


def test_config_models_keep_strict_unknown_field_check() -> None:
    with pytest.raises(ValidationError, match="unknown"):
        IngestConfig.model_validate({"unknown": True})
