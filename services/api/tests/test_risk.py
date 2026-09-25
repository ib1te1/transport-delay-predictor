import pytest

from app.config import RiskConfig
from app.risk import classify_risk


@pytest.mark.parametrize(
    ("prediction_s", "level"),
    [
        (-1000.0, "yellow"),
        (-60.5, "yellow"),
        (-60.0, "green"),
        (0.0, "green"),
        (120.0, "green"),
        (120.5, "yellow"),
        (300.0, "yellow"),
        (300.5, "red"),
    ],
)
def test_classify_risk_with_default_thresholds(prediction_s: float, level: str) -> None:
    assert classify_risk(prediction_s, RiskConfig()) == level


def test_classify_risk_follows_configured_thresholds() -> None:
    config = RiskConfig(green=(-30.0, 60.0), red_above=180.0)

    assert classify_risk(-31.0, config) == "yellow"
    assert classify_risk(61.0, config) == "yellow"
    assert classify_risk(181.0, config) == "red"
