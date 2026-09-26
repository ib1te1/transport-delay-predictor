"""Forecasts on the live stream.

:class:`LivePredictor` keeps the last hours of telemetry of every vehicle and on request
forecasts the delay at the stop the bus should reach in 10-15 minutes, the same way the
offline points are built: the same :func:`busdelay.features.point_features`, the same model.
The only difference is ``cur_dev_s``: on the stream nobody gives it, so it is estimated from
GPS with :func:`busdelay.features.estimate_cur_dev`.

When the telemetry of a vehicle stops coming, forecasts go on from the plan and the last
known state and are marked as degraded.
"""

import time
from collections import deque
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .explain import main_reason, reason_contributions
from .features import HISTORY_S, Track, estimate_cur_dev, point_features
from .model import DelayModel
from .schedule import VehiclePlan, prepare_plan, split_by_vehicle

# No fix for this long - the forecast is marked "no_signal".
STALE_AFTER_S = 5 * 60

# Risk colours for the dashboard, by the probability of being more than 2 minutes late.
RISK_LEVELS = ((0.6, "red"), (0.3, "yellow"))


def risk_level(late_prob: float) -> str:
    """Risk colour by the probability of being late, see ``RISK_LEVELS``."""
    for threshold, colour in RISK_LEVELS:
        if late_prob >= threshold:
            return colour
    return "green"


@dataclass
class Forecast:
    """One live forecast. Times in seconds since the epoch, the interval from CV residuals."""

    tr_id: int
    as_of: float
    target_stop_id: int
    target_planned: float
    delay_s: float
    delay_lo_s: float
    delay_hi_s: float
    late_prob: float
    risk: str
    reason: str
    reason_text: str
    reason_s: float
    cur_dev_s: float
    degraded: str | None


class LivePredictor:
    """Telemetry buffers plus the model. Not thread safe, one per worker."""

    def __init__(self, model: DelayModel, plan: pd.DataFrame, history_s: float = HISTORY_S):
        self.model = model
        self.plans: dict[int, VehiclePlan] = split_by_vehicle(prepare_plan(plan))
        self.history_s = history_s
        self._fixes: dict[int, deque] = {}

    def add_fix(self, tr_id: int, t: float, lat: float, lon: float, speed: float, ok: bool):
        """Store one telemetry point. Points must come in time order per vehicle."""
        fixes = self._fixes.setdefault(int(tr_id), deque())
        if fixes and t < fixes[-1][0]:
            return  # late duplicate, the stream already moved on
        fixes.append((float(t), bool(ok), float(lat), float(lon), float(speed)))
        while fixes and fixes[0][0] < t - self.history_s:
            fixes.popleft()

    def track(self, tr_id: int, now: float) -> Track:
        """The vehicle's buffered fixes up to ``now``."""
        fixes = self._fixes.get(int(tr_id))
        if not fixes:
            return Track.empty()
        t, ok, lat, lon, speed = (np.array(column) for column in zip(*fixes, strict=True))
        return Track(t, ok.astype(bool), lat, lon, speed).upto(now)

    def forecast(self, tr_id: int, now: float) -> Forecast | None:
        """Forecast for one vehicle at ``now``. None if it has no stop 10-15 minutes ahead."""
        result = self.forecast_many([tr_id], now)
        return result[0] if result else None

    def forecast_all(self, now: float) -> list[Forecast]:
        """Forecasts for every vehicle in the plan that has a target stop."""
        return self.forecast_many(list(self.plans), now)

    def forecast_many(self, tr_ids, now: float) -> list[Forecast]:
        """Forecasts for several vehicles. The model and SHAP run once for all of them,
        which is several times cheaper than one call per vehicle."""
        rows, context = [], []
        for tr_id in tr_ids:
            plan = self.plans.get(int(tr_id))
            if plan is None:
                continue
            target = plan.pick_target(now)
            if target < 0:
                continue
            track = self.track(tr_id, now)
            cur_dev = estimate_cur_dev(plan, track, now)
            rows.append(point_features(plan, track, now, plan.stop_id[target], cur_dev))
            degraded = None
            if not len(track) or now - track.t[-1] > STALE_AFTER_S:
                degraded = "no_signal"
            elif np.isnan(cur_dev):
                degraded = "no_position"
            context.append((int(tr_id), plan, target, cur_dev, degraded))
        if not rows:
            return []

        table = pd.DataFrame(rows)
        table["tr_id"] = [tr_id for tr_id, *_ in context]
        predicted = self.model.predict(table)
        reasons = main_reason(reason_contributions(self.model, table))

        result = []
        for i, (tr_id, plan, target, cur_dev, degraded) in enumerate(context):
            p = predicted.iloc[i]
            late_prob = float(p.get("late_prob", np.nan))
            result.append(
                Forecast(
                    tr_id=tr_id,
                    as_of=float(now),
                    target_stop_id=int(plan.stop_id[target]),
                    target_planned=float(plan.t_plan[target]),
                    delay_s=float(p["delay_s"]),
                    delay_lo_s=float(p["delay_lo_s"]),
                    delay_hi_s=float(p["delay_hi_s"]),
                    late_prob=late_prob,
                    risk=risk_level(late_prob) if not np.isnan(late_prob) else "unknown",
                    reason=str(reasons["reason"].iloc[i]),
                    reason_text=str(reasons["reason_text"].iloc[i]),
                    reason_s=float(reasons["reason_s"].iloc[i]),
                    cur_dev_s=float(cur_dev),
                    degraded=degraded,
                )
            )
        return result


def benchmark(predictor: LivePredictor, moments) -> dict[str, float]:
    """Time of one full cycle (all vehicles) and per forecast, in milliseconds."""
    cycles, forecasts = [], 0
    for now in moments:
        started = time.perf_counter()
        forecasts += len(predictor.forecast_all(now))
        cycles.append((time.perf_counter() - started) * 1000)
    cycles = np.array(cycles)
    return {
        "cycles": int(len(cycles)),
        "forecasts": int(forecasts),
        "cycle_p50_ms": float(np.percentile(cycles, 50)),
        "cycle_max_ms": float(cycles.max()),
        "per_forecast_ms": float(cycles.sum() / max(forecasts, 1)),
    }
