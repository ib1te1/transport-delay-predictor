"""Telemetry outages: points where the position stopped coming some minutes before ``T``.

On the stream a vehicle may go silent (the link is down) or keep reporting without a
position (the GPS is lost). The api marks a vehicle silent for 2 minutes as stale and drops
it after 15 (``api.stale_after_sec``, ``api.drop_after_sec``); a lost GPS it does not see.

The model gets no special treatment then: it forecasts from the last known state.
:func:`evaluate` measures how much that costs, on test points with the last minutes of
telemetry cut, next to the baselines. :func:`outage_points` makes such copies of the
labeled points for training (``features --outage``, ``train --outage-weight``); on the
cross-validation they did not help. Numbers are in docs/specs/ml-model.md.
"""

import numpy as np
import pandas as pd

from .baselines import SplitLinear
from .data import Part, is_synthetic
from .features import build_features
from .folds import block_ids
from .metrics import paired_interval
from .model import DelayModel

KINDS = ("silence", "no_position")
# Outages of the copies: from a minute up to api.drop_after_sec, after that the api stops
# asking for the vehicle.
GAP_S = (60.0, 900.0)
COPIES = 2
# A test point is used only if its last position is at most this old: the outage measured
# is then the simulated one.
FRESH_S = 30.0


def outage_points(points: pd.DataFrame, copies: int = COPIES, seed: int = 0) -> pd.DataFrame:
    """``copies`` of every real point, each with a random outage in ``gap_s`` and ``outage``.

    The gap is uniform in :data:`GAP_S`, the kind is one of :data:`KINDS` with equal odds.
    Pass the result to :func:`busdelay.features.build_features`.
    """
    real = points[~is_synthetic(points["tr_id"])]
    rng = np.random.default_rng(seed)
    return pd.concat(
        [
            real.assign(
                gap_s=rng.uniform(*GAP_S, len(real)),
                outage=rng.choice(KINDS, len(real)),
            )
            for _ in range(copies)
        ],
        ignore_index=True,
    )


def evaluate(
    models: dict[str, DelayModel], part: Part, minutes=(0, 2, 5, 10, 15), kinds=KINDS
) -> pd.DataFrame:
    """MAE of the models on ``part`` with the last ``minutes`` of telemetry cut.

    ``cur_dev_s`` is estimated from GPS, as the service does it. Next to the models are the
    two lines of the first model that has them, persistence and zero, all fed with the same
    estimate. The last model is compared with the two lines and with the first model:
    ``- <other>`` is the mean difference of absolute errors, ``- <other> lo`` and ``hi`` its
    90% bootstrap interval over runs of points. Only real points with fresh telemetry
    (:data:`FRESH_S`) are used.
    """
    points = part.points[~is_synthetic(part.points["tr_id"])].reset_index(drop=True)
    fresh = build_features(points, part.plan, part.telemetry, hint="gps")["tel_age_s"]
    points = points[(fresh <= FRESH_S).to_numpy()].reset_index(drop=True)
    if points.empty:
        raise ValueError(f"no points with telemetry fresher than {FRESH_S:.0f} s in {part.name}")
    y = points["target"].to_numpy(dtype=float)
    blocks = block_ids(points).to_numpy()
    lines = next((m.lines for m in models.values() if m.lines is not None), None)
    names = list(models)

    cases = [("none", 0)] + [(k, m) for k in kinds for m in minutes if m > 0]
    rows = []
    for kind, gap in cases:
        cut = points.assign(gap_s=gap * 60.0, outage=kind)
        table = build_features(cut, part.plan, part.telemetry, hint="gps")
        cur = table["cur_dev_s"].fillna(0.0).to_numpy()
        predicted = {"persistence": cur, "zero": np.zeros(len(table))}
        if lines is not None:
            base = SplitLinear()
            base.lines = lines
            predicted["split_linear"] = base.predict(table)
        for name, model in models.items():
            predicted[name] = model.predict(table)["delay_s"].to_numpy()
        error = {name: np.abs(y - p) for name, p in predicted.items()}
        row = {"outage": kind, "minutes": gap, "points": len(y)}
        row.update({name: e.mean() for name, e in error.items()})
        others = [n for n in ("split_linear", names[0]) if n in error and n != names[-1]]
        for other in others:
            difference = error[names[-1]] - error[other]
            row[f"- {other}"] = difference.mean()
            row[f"- {other} lo"], row[f"- {other} hi"] = paired_interval(difference, blocks)
        rows.append(row)
    return pd.DataFrame(rows)
