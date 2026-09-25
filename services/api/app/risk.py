"""Risk level of a vehicle from its predicted delay at the target stop."""

from app.config import RiskConfig
from app.models import RiskLevel


def classify_risk(prediction_s: float, config: RiskConfig) -> RiskLevel:
    """Green inside ``config.green``, red above ``config.red_above``, yellow otherwise.

    Running early past green's lower bound is yellow, not green: an early
    departure breaks the headway just as a late one does.
    """
    low, high = config.green
    if low <= prediction_s <= high:
        return "green"
    if prediction_s > config.red_above:
        return "red"
    return "yellow"
