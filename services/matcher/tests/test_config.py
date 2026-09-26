from pathlib import Path

import pytest

from app.config import MatcherConfig, load_matcher_settings
from common.config import load_section


def test_project_assumptions_load():
    path = Path(__file__).resolve().parents[3] / "config" / "assumptions.yaml"
    settings = load_matcher_settings(path)
    assert settings.stop_radius_m == 50
    assert settings.visit_time_tolerance_sec == 300


def test_unknown_matcher_assumption_fails(tmp_path):
    path = tmp_path / "assumptions.yaml"
    path.write_text("matcher:\n  stop_radius_m: 50\n  typo: 1\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_matcher_settings(path)


def test_matcher_config_defaults_and_bounds():
    config = MatcherConfig()
    assert (config.batch_size, config.block_ms) == (500, 1000)
    path = Path(__file__).resolve().parents[3] / "config" / "system.yaml"
    assert load_section(path, "matcher", MatcherConfig) == config
    for bad in ({"batch_size": 0}, {"batch_size": 10_001}, {"block_ms": 0}, {"block_ms": 60_001}):
        with pytest.raises(ValueError):
            MatcherConfig(**bad)
