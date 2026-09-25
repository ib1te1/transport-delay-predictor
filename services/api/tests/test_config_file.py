from app.config import ApiConfig, SeedConfig
from common.config import REPO_ROOT, load_section

SYSTEM_YAML = REPO_ROOT / "config" / "system.yaml"


def test_repository_config_has_valid_api_section():
    config = load_section(SYSTEM_YAML, "api", ApiConfig)
    assert config.risk.green == (-60.0, 120.0)
    assert config.request.telemetry_window_sec > 0


def test_repository_config_has_valid_seed_section():
    assert load_section(SYSTEM_YAML, "seed", SeedConfig).period == "test"
