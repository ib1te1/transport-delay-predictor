"""YAML config and environment settings for the services.

Services start from different places — /app in a container, the service
folder in a local venv — so nothing here depends on the working
directory. Defaults resolve from where this package lives, which with
the editable install in a venv is the repository itself. In a container
compose sets every value, and these defaults are not used. Outside an
editable install — that is, inside the images — this default resolves
outside the repository, so the compose environment variables are not
optional there.
"""

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

# packages/common/src/common/config.py -> repository root.
REPO_ROOT = Path(__file__).resolve().parents[4]


class StrictModel(BaseModel):
    """Base for config models: an unknown key is an error, not a no-op."""

    model_config = ConfigDict(extra="forbid")


class ConfigError(ValueError):
    """A config file that is missing, malformed or fails validation."""


def _read_yaml(path: Path) -> dict:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"{path}: {exc}") from exc
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    return raw


def _validate[M: BaseModel](path: Path, raw: object, model: type[M], prefix: str = "") -> M:
    if model.model_config.get("extra") != "forbid":
        raise TypeError(f"{model.__name__} must forbid extra keys: derive it from StrictModel")
    try:
        return model.model_validate(raw if raw is not None else {})
    except ValidationError as exc:
        details = "\n".join(
            f"  {prefix}{'.'.join(map(str, error['loc'])) or '<root>'}: {error['msg']}"
            for error in exc.errors()
        )
        raise ConfigError(f"{path}: invalid config\n{details}") from exc


def load_yaml[M: BaseModel](path: Path | str, model: type[M]) -> M:
    """Parse a YAML file and validate it; errors name the file and field path.

    Nested sections should derive from StrictModel as well, or a typo
    inside them is silently ignored.
    """
    path = Path(path)
    return _validate(path, _read_yaml(path), model)


def load_section[M: BaseModel](path: Path | str, section: str, model: type[M]) -> M:
    """Validate one top-level section of a shared config file.

    Each service reads only its own section, so a track can add keys to
    its section without every other service learning about them. A
    missing section is an error: it is more likely a typo than an intent.
    """
    path = Path(path)
    raw = _read_yaml(path)
    if section not in raw:
        raise ConfigError(f"{path}: no section {section!r}")
    return _validate(path, raw[section], model, prefix=f"{section}.")


class ServiceSettings(BaseSettings):
    """Connection settings every service reads at startup.

    A missing required variable fails startup with the variable's name.
    Locally the repository's .env fills in what the shell does not set.
    """

    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    database_url: str
    redis_url: str
    predictor_url: str | None = None
    config_path: Path = REPO_ROOT / "config" / "system.yaml"
    data_dir: Path = REPO_ROOT / "data" / "dataset"
