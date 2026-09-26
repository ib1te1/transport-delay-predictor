"""The model behind ``POST /predict``, or the baseline when there is no model.

Features, model and reasons are :mod:`busdelay` code, the same that builds the submission.
This module only translates the contract into its queries and back.
"""

import logging
import math
import threading
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from app.config import PredictorConfig
from busdelay.inference import Answer, Forecaster, Query
from busdelay.model import REGRESSOR_FILE, DelayModel
from contracts import PredictRequest, PredictResponse, ReasonCode

log = logging.getLogger(__name__)

BASELINE_VERSION = "baseline"

# busdelay reasons -> contract codes. The contract has no code for the layover at the
# terminal and for the timetable itself; those stay "unknown".
REASON_CODES = {
    "carried_delay": ReasonCode.accumulated_delay,
    "slow_traffic": ReasonCode.slow_approach,
    "long_dwell": ReasonCode.long_dwell,
    "chronic_segment": ReasonCode.low_speed_segment,
    "signal": ReasonCode.stale_telemetry,
    "layover": ReasonCode.unknown,
    "timetable": ReasonCode.unknown,
}


class Predictor(Protocol):
    version: str

    def predict(self, requests: list[PredictRequest]) -> list[PredictResponse]: ...


def baseline_response(request: PredictRequest) -> PredictResponse:
    """The organisers' baseline: the current deviation persists."""
    return PredictResponse(
        sample_id=request.sample_id,
        prediction_s=request.cur_dev_s if request.cur_dev_s is not None else 0.0,
        p_late=None,
        reasons=[ReasonCode.unknown],
        model_version=BASELINE_VERSION,
    )


class BaselinePredictor:
    version = BASELINE_VERSION

    def __init__(self, why: str = "no model configured"):
        self.why = why

    def predict(self, requests: list[PredictRequest]) -> list[PredictResponse]:
        return [baseline_response(r) for r in requests]


class ModelPredictor:
    def __init__(self, model_dir: Path, config: PredictorConfig):
        self.model = DelayModel.load(model_dir)
        hint = "gps" if config.cur_dev_source == "gps" else "given"
        self.forecaster = Forecaster(self.model, hint=hint)
        self.version = f"{model_dir.name}@{self.model.meta.get('created', 'unknown')}"
        self.stale_after_s = float(config.stale_after_sec)
        # CatBoost may be called from several worker threads at once
        self._lock = threading.Lock()

    def predict(self, requests: list[PredictRequest]) -> list[PredictResponse]:
        queries = [to_query(r) for r in requests]
        with self._lock:
            answers = self.forecaster.predict(queries)
        responses = []
        for request, answer in zip(requests, answers, strict=True):
            if answer.fallback is not None:
                log.warning(
                    "%s: baseline instead of the model: %s", answer.sample_id, answer.fallback
                )
                responses.append(baseline_response(request))
            else:
                responses.append(to_response(answer, self.version, self.stale_after_s))
        return responses


def load_predictor(model_dir: Path | None, config: PredictorConfig) -> Predictor:
    """The model from ``model_dir``; the baseline if there is none or it does not load.

    A broken model must not take the service down: the api then gets the baseline, and
    ``GET /model`` says why.
    """
    if model_dir is None or not (model_dir / REGRESSOR_FILE).exists():
        log.warning("no model in %s, answering with the baseline", model_dir)
        return BaselinePredictor(f"no model in {model_dir}")
    try:
        predictor = ModelPredictor(model_dir, config)
    except Exception as exc:  # any failure to load means the same fallback
        log.exception("model in %s does not load, answering with the baseline", model_dir)
        return BaselinePredictor(f"model in {model_dir} does not load: {exc}")
    log.info("model %s loaded, cur_dev_s from %s", predictor.version, config.cur_dev_source)
    return predictor


def _floats(values) -> np.ndarray:
    # None becomes NaN
    return np.array(values, dtype=float)


def to_query(request: PredictRequest) -> Query:
    """The contract's request as a busdelay query: times in seconds, missing values NaN.

    Fact times in the schedule are left out: the model is built from the plan only.
    """
    schedule, telemetry = request.schedule, request.telemetry
    plan = pd.DataFrame(
        {
            "stop_id": np.array([s.stop_id for s in schedule], dtype="int64"),
            "t_plan": _floats([s.time_plan.timestamp() for s in schedule]),
            "lat": _floats([s.lat for s in schedule]),
            "lon": _floats([s.lon for s in schedule]),
        }
    )
    fixes = pd.DataFrame(
        {
            "t": _floats([p.event_time.timestamp() for p in telemetry]),
            "location_valid": np.array([p.location_valid for p in telemetry], dtype=bool),
            "lat": _floats([p.lat for p in telemetry]),
            "lon": _floats([p.lon for p in telemetry]),
            "speed": _floats([p.speed_kmh for p in telemetry]),
        }
    )
    return Query(
        sample_id=request.sample_id,
        tr_id=request.tr_id,
        T=request.T.timestamp(),
        target_stop_id=request.target_stop_id,
        cur_dev_s=math.nan if request.cur_dev_s is None else float(request.cur_dev_s),
        plan=plan,
        fixes=fixes,
    )


def reason_codes(reasons: list[str]) -> list[ReasonCode]:
    """Contract codes in the model's order, each once; ``unknown`` only if nothing else."""
    codes = list(dict.fromkeys(REASON_CODES[r] for r in reasons))
    known = [c for c in codes if c is not ReasonCode.unknown]
    return known or [ReasonCode.unknown]


def to_response(answer: Answer, version: str, stale_after_s: float) -> PredictResponse:
    """The contract's answer. A forecast from an old position, or from none at all, also
    gets ``stale_telemetry``: the api does not see a vehicle that reports without a position.
    """
    reasons = list(answer.reasons)
    # NaN, no position at all, is not <= either
    if not answer.position_age_s <= stale_after_s:
        reasons.append("signal")
    return PredictResponse(
        sample_id=answer.sample_id,
        prediction_s=answer.delay_s,
        p_late=None if math.isnan(answer.late_prob) else answer.late_prob,
        reasons=reason_codes(reasons),
        model_version=version,
    )
