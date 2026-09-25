import pytest
from pydantic import ValidationError

from app.config import ApiConfig, RiskConfig, SeedConfig
from common.config import REPO_ROOT, load_section

SYSTEM_YAML = REPO_ROOT / "config" / "system.yaml"


def test_repository_config_has_valid_api_section():
    config = load_section(SYSTEM_YAML, "api", ApiConfig)
    assert config.risk.green == (-60.0, 120.0)
    assert config.request.telemetry_window_sec > 0


def test_repository_config_has_valid_seed_section():
    assert load_section(SYSTEM_YAML, "seed", SeedConfig).period == "test"


def test_risk_config_rejects_a_swapped_green_pair():
    with pytest.raises(ValidationError, match="risk.green"):
        RiskConfig(green=(120.0, -60.0), red_above=300.0)


def test_risk_config_rejects_red_above_below_green():
    with pytest.raises(ValidationError, match="red_above"):
        RiskConfig(green=(-60.0, 120.0), red_above=100.0)
