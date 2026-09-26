from pathlib import Path

import pytest

from app.config import load_matcher_settings


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
