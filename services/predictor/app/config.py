"""The predictor section of config/system.yaml and the environment settings."""

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from common.config import REPO_ROOT, StrictModel


class PredictorConfig(StrictModel):
    # gps - estimated from the request's telemetry, the way the model saw it in training;
    # the value sent with the request only fills in when the estimate fails.
    # request - as sent.
    cur_dev_source: Literal["gps", "request"] = "gps"
    # A forecast whose last valid position is older than this, or missing, gets the
    # stale_telemetry reason. The api flags only vehicles that went silent; one that keeps
    # reporting without a position it does not notice.
    stale_after_sec: int = Field(default=120, gt=0)


class PredictorSettings(BaseSettings):
    """Where the model and the config are. Compose sets both in the container."""

    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        # "model_dir" would otherwise clash with pydantic's own model_ names
        protected_namespaces=(),
    )

    model_dir: Path | None = REPO_ROOT / "models" / "current"
    config_path: Path = REPO_ROOT / "config" / "system.yaml"
