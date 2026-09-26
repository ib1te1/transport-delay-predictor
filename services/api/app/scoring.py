"""One scoring tick: candidates, predictor or fallback, prediction rows."""

import logging
from collections.abc import Awaitable, Callable, Sequence
from datetime import datetime

from app.config import ApiConfig
from app.models import PredictionRow
from app.planner import Candidate, PlanIndex, plan_tick
from app.predictor_client import PredictorError
from app.risk import classify_risk
from app.state import FleetState
from contracts import PredictRequest, PredictResponse, ReasonCode

log = logging.getLogger(__name__)

FALLBACK_MODEL_VERSION = "fallback"

type Predict = Callable[[Sequence[PredictRequest]], Awaitable[list[PredictResponse]]]
type Save = Callable[[list[PredictionRow]], Awaitable[list[PredictionRow]]]
type Sink = Callable[[list[PredictionRow]], Awaitable[None]]


def fallback_response(request: PredictRequest) -> PredictResponse:
    """The organizers' baseline in place of the model: the current deviation persists."""
    return PredictResponse(
        sample_id=request.sample_id,
        prediction_s=request.cur_dev_s if request.cur_dev_s is not None else 0.0,
        p_late=None,
        reasons=[ReasonCode.unknown],
        model_version=FALLBACK_MODEL_VERSION,
    )


def degraded_reason(*, fallback: bool, stale: bool, warming_up: bool) -> str | None:
    """The single reason a row records, by precedence: predictor, telemetry age, warm-up."""
    if fallback:
        return "predictor_unavailable"
    if stale:
        return "stale_telemetry"
    if warming_up:
        return "warming_up"
    return None


def to_row(
    candidate: Candidate, answer: PredictResponse, *, fallback: bool, config: ApiConfig
) -> PredictionRow:
    """The ``predictions`` row for one answer, with risk level and degradation flags."""
    request = candidate.request
    reasons = [r.value for r in answer.reasons]
    if candidate.stale and ReasonCode.stale_telemetry.value not in reasons:
        reasons.append(ReasonCode.stale_telemetry.value)
    reason = degraded_reason(
        fallback=fallback, stale=candidate.stale, warming_up=candidate.warming_up
    )
    return PredictionRow(
        sample_id=request.sample_id,
        tr_id=request.tr_id,
        t=request.T,
        target_stop_id=request.target_stop_id,
        target_time_begin=request.target_time_begin,
        cur_dev_s=request.cur_dev_s,
        prediction_s=answer.prediction_s,
        p_late=answer.p_late,
        reasons=reasons,
        risk_level=classify_risk(answer.prediction_s, config.risk),
        degraded=reason is not None,
        degraded_reason=reason,
        model_version=answer.model_version,
        actual_delay_s=None,
        abs_error_s=None,
    )


async def run_tick(
    t: datetime,
    state: FleetState,
    plan: PlanIndex,
    config: ApiConfig,
    *,
    predict: Predict,
    save: Save,
    publish: Sink,
) -> list[PredictionRow]:
    """Score every candidate at ``t`` in one batch.

    If predictor fails, the whole batch falls back to the current
    deviation; the next tick tries predictor again. Rows are saved before
    they are published, so a reader of the stream finds them in the
    table; a row whose ``sample_id`` was already stored is not published
    again.
    """
    state.prune()
    candidates = plan_tick(state, plan, t, config)
    if not candidates:
        return []
    requests = [c.request for c in candidates]
    fallback = False
    try:
        answers = await predict(requests)
    except PredictorError as exc:
        log.warning("predictor unavailable, falling back for %d request(s): %s", len(requests), exc)
        answers = [fallback_response(r) for r in requests]
        fallback = True
    rows = [
        to_row(c, a, fallback=fallback, config=config)
        for c, a in zip(candidates, answers, strict=True)
    ]
    saved = await save(rows)
    if len(saved) < len(rows):
        log.info("%d prediction row(s) already stored; not published again", len(rows) - len(saved))
    if saved:
        await publish(saved)
    return rows
