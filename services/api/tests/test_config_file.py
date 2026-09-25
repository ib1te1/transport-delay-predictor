import pytest
from pydantic import ValidationError

from app.config import ApiConfig, RequestConfig, RiskConfig, SeedConfig
from common.config import REPO_ROOT, load_dataset_config, load_section

SYSTEM_YAML = REPO_ROOT / "config" / "system.yaml"


def test_repository_config_has_valid_api_section():
    config = load_section(SYSTEM_YAML, "api", ApiConfig)
    assert config.risk.green == (-60.0, 120.0)
    assert config.request.telemetry_window_sec > 0


def test_repository_config_has_valid_seed_section():
    assert load_section(SYSTEM_YAML, "seed", SeedConfig).period == "test"


def test_repository_config_has_valid_dataset_section():
    assert load_dataset_config(SYSTEM_YAML).source_timezone == "UTC"


def test_risk_config_rejects_a_swapped_green_pair():
    with pytest.raises(ValidationError, match="risk.green"):
        RiskConfig(green=(120.0, -60.0), red_above=300.0)


def test_risk_config_rejects_red_above_below_green():
    with pytest.raises(ValidationError, match="red_above"):
        RiskConfig(green=(-60.0, 120.0), red_above=100.0)


def test_api_config_rejects_stale_after_longer_than_drop_after():
    with pytest.raises(ValidationError, match="got 1000, 900, 1800"):
        ApiConfig(stale_after_sec=1000, drop_after_sec=900)


def test_api_config_rejects_drop_after_longer_than_the_telemetry_window():
    with pytest.raises(ValidationError, match="got 120, 2000, 1800"):
        ApiConfig(drop_after_sec=2000)


def test_api_config_accepts_equal_time_limits():
    config = ApiConfig(
        stale_after_sec=900,
        drop_after_sec=900,
        request=RequestConfig(telemetry_window_sec=900),
    )
    assert config.drop_after_sec == 900
