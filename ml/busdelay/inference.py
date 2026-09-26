"""Forecasts for self-contained requests: the plan and the telemetry come with each point.

This is what the predictor service runs. For every vehicle the api sends its planned stops
and its telemetry up to ``T``; features, model and reasons are the same code as offline
(:func:`busdelay.features.point_features`), so the service and the submission cannot drift
apart.

What the model needs in a request (measured on test, docs/specs/ml-model.md):

* telemetry for :data:`busdelay.features.HISTORY_S` (150 minutes) before ``T``: with 30
  minutes the previous loop is not seen and MAE grows by 4 s;
* the plan from the start of the vehicle's day: trips are counted from there. Stops after
  the target are not needed.

``cur_dev_s`` is estimated from GPS by default (``hint="gps"``), as the model saw it in
training; the value sent with the request is used only if the estimate fails. With
``hint="given"`` the sent value is used as is: that is the offline submission, where the
organisers give it.

When the position stops coming (the link is down or the GPS is lost), the model still
answers from the last known state (:mod:`busdelay.outage` measures what that costs). How old
the last position is goes back with the answer, so that the service can flag it.
"""

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from .data import clean_telemetry
from .explain import REASONS, reason_contributions
from .features import Track, estimate_cur_dev, point_features
from .model import DelayModel
from .schedule import VehiclePlan, prepare_plan, split_by_vehicle

HINTS = ("gps", "given")
PLAN_COLUMNS = ["stop_id", "t_plan", "lat", "lon"]
FIX_COLUMNS = ["t", "location_valid", "lat", "lon", "speed"]
# A reason is named only if it adds at least this much to the predicted delay.
REASON_MIN_S = 15.0
MAX_REASONS = 3


@dataclass
class Query:
    """One forecast point with everything the model may see. Times in seconds since the epoch.

    ``plan`` has ``stop_id, t_plan, lat, lon`` of this vehicle; ``fixes`` has
    ``t, location_valid, lat, lon, speed`` (km/h), missing values as NaN.
    """

    sample_id: str
    tr_id: int
    T: float
    target_stop_id: int
    cur_dev_s: float
    plan: pd.DataFrame
    fixes: pd.DataFrame


@dataclass
class Answer:
    """The forecast for one :class:`Query`."""

    sample_id: str
    delay_s: float
    # NaN when the model has no classifier or the point fell back to the baseline
    late_prob: float
    # keys of busdelay.explain.REASONS, the strongest push towards late first
    reasons: list[str]
    # the cur_dev_s the forecast was made with, NaN if none was known
    cur_dev_s: float
    # seconds from the last fix with a valid position to T, NaN if there is none
    position_age_s: float = np.nan
    # why the model could not be used for this point, None when it was
    fallback: str | None = None


def batch_inputs(queries: list[Query]) -> tuple[dict[int, VehiclePlan], dict[int, Track]]:
    """Plans and cleaned tracks of all queries, keyed by the query's position.

    Everything is prepared in one pass, with the position standing in for ``tr_id``: pandas
    costs per call much more than per row, and one pass per query took most of the time.
    The telemetry is cleaned the way the dataset is cleaned for training.
    """
    plans = [q.plan[PLAN_COLUMNS].assign(tr_id=i) for i, q in enumerate(queries)]
    plans = [p for p in plans if not p.empty]
    prepared = split_by_vehicle(prepare_plan(pd.concat(plans))) if plans else {}
    fixes = [q.fixes[FIX_COLUMNS].assign(tr_id=i) for i, q in enumerate(queries)]
    fixes = [f for f in fixes if not f.empty]
    tracks = {}
    if fixes:
        cleaned = clean_telemetry(pd.concat(fixes, ignore_index=True))
        tracks = {int(i): Track.from_frame(rows) for i, rows in cleaned.groupby("tr_id")}
    return (
        {i: replace(plan, tr_id=queries[i].tr_id) for i, plan in prepared.items()},
        tracks,
    )


class Forecaster:
    """The model behind the service. Stateless: every call gets everything it needs."""

    def __init__(self, model: DelayModel, hint: str = "gps"):
        if hint not in HINTS:
            raise ValueError(f"hint must be one of {', '.join(HINTS)}, got {hint!r}")
        self.model = model
        self.hint = hint

    def cur_dev(self, query: Query, plan: VehiclePlan, track: Track) -> float:
        """The ``cur_dev_s`` the forecast is made with: sent or estimated, by ``hint``."""
        sent = float(query.cur_dev_s)
        if self.hint == "given":
            return sent
        estimated = estimate_cur_dev(plan, track, query.T)
        return sent if np.isnan(estimated) else estimated

    def predict(self, queries: list[Query]) -> list[Answer]:
        """Answers in the order of ``queries``. The model and SHAP run once for the batch.

        A point whose features cannot be built (its target is not in the plan or is planned
        before ``T``) gets the persistence answer with ``fallback`` set instead of failing
        the whole batch.
        """
        answers: list[Answer | None] = [None] * len(queries)
        plans, tracks = batch_inputs(queries)
        rows, index, used, ages = [], [], [], []
        for i, query in enumerate(queries):
            track = tracks.get(i, Track.empty()).upto(query.T)
            seen = track.t[track.ok]
            age = query.T - seen[-1] if seen.size else np.nan
            try:
                if i not in plans:
                    raise KeyError(f"no planned stops for vehicle {query.tr_id}")
                plan = plans[i]
                cur = self.cur_dev(query, plan, track)
                rows.append(point_features(plan, track, query.T, query.target_stop_id, cur))
            except (KeyError, ValueError) as exc:
                sent = float(query.cur_dev_s)
                answers[i] = Answer(
                    sample_id=query.sample_id,
                    delay_s=0.0 if np.isnan(sent) else sent,
                    late_prob=np.nan,
                    reasons=[],
                    cur_dev_s=sent,
                    position_age_s=age,
                    fallback=str(exc),
                )
                continue
            index.append(i)
            used.append(cur)
            ages.append(age)

        if rows:
            table = pd.DataFrame(rows)
            predicted = self.model.predict(table)
            contributions = reason_contributions(self.model, table)[list(REASONS)].to_numpy()
            late = (
                predicted["late_prob"].to_numpy()
                if "late_prob" in predicted
                else np.full(len(table), np.nan)
            )
            for k, i in enumerate(index):
                answers[i] = Answer(
                    sample_id=queries[i].sample_id,
                    delay_s=float(predicted["delay_s"].iloc[k]),
                    late_prob=float(late[k]),
                    reasons=top_reasons(contributions[k]),
                    cur_dev_s=float(used[k]),
                    position_age_s=float(ages[k]),
                )
        return answers


def queries_for_points(
    points: pd.DataFrame, plan: pd.DataFrame, telemetry: pd.DataFrame
) -> list[Query]:
    """Queries for dataset points, the way the service gets them: the vehicle's whole plan
    and its telemetry up to ``T``. Frames as :func:`busdelay.data.load_part` returns them."""
    plans = {int(tr_id): rows for tr_id, rows in plan.groupby("tr_id")}
    fixes = {
        int(tr_id): rows.rename(columns={"ok": "location_valid"}).reset_index(drop=True)
        for tr_id, rows in telemetry.groupby("tr_id")
    }
    empty = pd.DataFrame(columns=["t", "location_valid", "lat", "lon", "speed"])
    queries = []
    for p in points.itertuples(index=False):
        track = fixes.get(int(p.tr_id), empty)
        queries.append(
            Query(
                sample_id=p.sample_id,
                tr_id=int(p.tr_id),
                T=float(p.T),
                target_stop_id=int(p.target_stop_id),
                cur_dev_s=float(p.cur_dev_s),
                plan=plans[int(p.tr_id)],
                fixes=track[track["t"] <= p.T],
            )
        )
    return queries


def top_reasons(contribution: np.ndarray) -> list[str]:
    """Reasons that add at least :data:`REASON_MIN_S` to the delay, the largest first."""
    keys = list(REASONS)
    order = np.argsort(-contribution)
    return [keys[j] for j in order[:MAX_REASONS] if contribution[j] >= REASON_MIN_S]
