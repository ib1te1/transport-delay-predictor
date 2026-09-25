from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
from pydantic import BaseModel, ValidationError

from common.config import (
    REPO_ROOT,
    ConfigError,
    DatasetConfig,
    ServiceSettings,
    StrictModel,
    load_dataset_config,
    load_section,
    load_yaml,
)


class Matcher(StrictModel):
    stop_radius_m: int
    stopped_speed_kmh: float


class Toy(StrictModel):
    name: str
    matcher: Matcher


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "toy.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_valid_file_loads(tmp_path: Path) -> None:
    path = write(tmp_path, "name: toy\nmatcher:\n  stop_radius_m: 50\n  stopped_speed_kmh: 5\n")

    assert load_yaml(path, Toy) == Toy(
        name="toy", matcher=Matcher(stop_radius_m=50, stopped_speed_kmh=5)
    )


def test_unknown_key_fails_with_its_path(tmp_path: Path) -> None:
    path = write(
        tmp_path,
        "name: toy\nmatcher:\n  stop_radius_m: 50\n  stopped_speed_kmh: 5\n  stop_raduis_m: 60\n",
    )

    with pytest.raises(ConfigError, match=r"matcher\.stop_raduis_m"):
        load_yaml(path, Toy)


def test_wrong_type_fails_with_its_path(tmp_path: Path) -> None:
    path = write(tmp_path, "name: toy\nmatcher:\n  stop_radius_m: far\n  stopped_speed_kmh: 5\n")

    with pytest.raises(ConfigError, match=r"matcher\.stop_radius_m") as caught:
        load_yaml(path, Toy)
    assert str(path) in str(caught.value)


def test_missing_file_is_a_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        load_yaml(tmp_path / "absent.yaml", Toy)


def test_model_that_allows_extra_keys_is_refused(tmp_path: Path) -> None:
    class Loose(BaseModel):
        name: str

    with pytest.raises(TypeError, match="StrictModel"):
        load_yaml(write(tmp_path, "name: x\n"), Loose)


def test_settings_read_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql://db")
    monkeypatch.setenv("REDIS_URL", "redis://bus")
    monkeypatch.delenv("PREDICTOR_URL", raising=False)
    monkeypatch.delenv("CONFIG_PATH", raising=False)

    settings = ServiceSettings(_env_file=None)

    assert settings.database_url == "postgresql://db"
    assert settings.redis_url == "redis://bus"
    assert settings.predictor_url is None
    assert settings.config_path == REPO_ROOT / "config" / "system.yaml"


def test_missing_required_variable_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("REDIS_URL", "redis://bus")

    with pytest.raises(ValidationError, match="database_url"):
        ServiceSettings(_env_file=None)


def test_config_path_can_be_overridden(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql://db")
    monkeypatch.setenv("REDIS_URL", "redis://bus")
    monkeypatch.setenv("CONFIG_PATH", "/config/system.yaml")

    assert ServiceSettings(_env_file=None).config_path == Path("/config/system.yaml")


def test_service_settings_defaults_point_into_the_repository(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql://x")
    monkeypatch.setenv("REDIS_URL", "redis://x")
    monkeypatch.delenv("CONFIG_PATH", raising=False)
    monkeypatch.delenv("DATA_DIR", raising=False)
    settings = ServiceSettings(_env_file=None)
    assert settings.config_path == REPO_ROOT / "config" / "system.yaml"
    assert settings.data_dir == REPO_ROOT / "data" / "dataset"
    assert settings.config_path.is_file()


def test_dataset_config_valid_zone_loads() -> None:
    config = DatasetConfig(source_timezone="Europe/Moscow")
    assert config.zone == ZoneInfo("Europe/Moscow")


def test_dataset_config_rejects_an_unknown_zone() -> None:
    with pytest.raises(ValidationError, match="Mars/Olympus"):
        DatasetConfig(source_timezone="Mars/Olympus")


def test_load_dataset_config_names_the_field_on_an_unknown_zone(tmp_path: Path) -> None:
    path = tmp_path / "system.yaml"
    path.write_text("dataset:\n  source_timezone: Mars/Olympus\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="dataset.source_timezone"):
        load_dataset_config(path)


class _Section(StrictModel):
    size: int = 1


def test_load_section_reads_only_its_section(tmp_path):
    path = tmp_path / "system.yaml"
    path.write_text("mine:\n  size: 3\nother:\n  anything: [1, 2]\n", encoding="utf-8")
    assert load_section(path, "mine", _Section).size == 3


def test_load_section_empty_section_uses_defaults(tmp_path):
    path = tmp_path / "system.yaml"
    path.write_text("mine: {}\n", encoding="utf-8")
    assert load_section(path, "mine", _Section).size == 1


def test_load_section_missing_section_is_an_error(tmp_path):
    path = tmp_path / "system.yaml"
    path.write_text("other: {}\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="no section 'mine'"):
        load_section(path, "mine", _Section)


def test_load_section_typo_names_the_field(tmp_path):
    path = tmp_path / "system.yaml"
    path.write_text("mine:\n  sise: 3\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="mine.sise"):
        load_section(path, "mine", _Section)
