"""Which vehicles get a prediction at T, and the request each one sends.

The rule mirrors the organizers' labels: the target is the first planned
stop 10 to 15 minutes ahead of T, and the request carries nothing that
happened after T.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta

from app.config import ApiConfig
from app.models import PlanStop
from app.state import FleetState
from contracts import (
    PredictRequest,
    ScheduledStop,
    StopEvent,
    TelemetryPoint,
    TelemetryRecord,
    make_sample_id,
)

HORIZON_BEGIN = timedelta(minutes=10)
HORIZON_END = timedelta(minutes=15)

type PlanIndex = dict[int, list[PlanStop]]
"""Planned stops per ``tr_id``, each list ordered by ``time_plan``."""


@dataclass(frozen=True)
class Candidate:
    """A request to score, with the input-quality flags the row will carry."""

    request: PredictRequest
    stale: bool
    warming_up: bool


def select_target(stops: list[PlanStop], t: datetime) -> PlanStop | None:
    """The first stop with ``time_plan`` in ``(t + 10 min, t + 15 min]``.

    ``stops`` must be ordered by ``time_plan``.
    """
    begin, end = t + HORIZON_BEGIN, t + HORIZON_END
    for stop in stops:
        if stop.time_plan > end:
            break
        if stop.time_plan > begin:
            return stop
    return None


def build_request(
    tr_id: int,
    t: datetime,
    target: PlanStop,
    stops: list[PlanStop],
    telemetry: list[TelemetryRecord],
    events: list[StopEvent],
    config: ApiConfig,
) -> PredictRequest:
    """Assemble what the model may see at ``t``; nothing after ``t`` gets in."""
    window_start = t - timedelta(seconds=config.request.telemetry_window_sec)
    schedule_start = t - timedelta(seconds=config.request.schedule_back_sec)
    passed = [e for e in events if e.time_fact <= t]
    fact_by_stop = {e.stop_id: e.time_fact for e in passed}
    latest = max(passed, key=lambda e: e.time_fact) if passed else None
    return PredictRequest(
        sample_id=make_sample_id(tr_id, t),
        tr_id=tr_id,
        T=t,
        target_stop_id=target.stop_id,
        target_time_begin=target.time_plan,
        cur_dev_s=latest.delay_s if latest else None,
        telemetry=[_point(r) for r in telemetry if window_start <= r.event_time <= t],
        schedule=[
            ScheduledStop(
                stop_id=s.stop_id,
                time_plan=s.time_plan,
                lat=s.lat,
                lon=s.lon,
                time_fact=fact_by_stop.get(s.stop_id),
            )
            for s in stops
            if schedule_start <= s.time_plan <= target.time_plan
        ],
    )


def plan_tick(
    state: FleetState, plan: PlanIndex, t: datetime, config: ApiConfig
) -> list[Candidate]:
    """One candidate per active vehicle that has a target stop at ``t``.

    A vehicle silent for longer than ``drop_after_sec`` is left out;
    longer than ``stale_after_sec``, it is predicted from its last state
    and flagged stale.
    """
    stale_after = timedelta(seconds=config.stale_after_sec)
    drop_after = timedelta(seconds=config.drop_after_sec)
    warming_up = state.warming_up(t)
    candidates = []
    for tr_id in state.vehicles():
        telemetry = [r for r in state.telemetry(tr_id) if r.event_time <= t]
        if not telemetry:
            continue
        silence = t - telemetry[-1].event_time
        if silence > drop_after:
            continue
        stops = plan.get(tr_id, [])
        target = select_target(stops, t)
        if target is None:
            continue
        request = build_request(
            tr_id, t, target, stops, telemetry, state.stop_events(tr_id), config
        )
        candidates.append(
            Candidate(request=request, stale=silence > stale_after, warming_up=warming_up)
        )
    return candidates


def _point(record: TelemetryRecord) -> TelemetryPoint:
    return TelemetryPoint(
        event_time=record.event_time,
        lat=record.lat,
        lon=record.lon,
        location_valid=record.location_valid,
        speed_kmh=record.speed_kmh,
        heading_deg=record.heading_deg,
    )
