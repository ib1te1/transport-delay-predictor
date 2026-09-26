"""The dashboard's views of vehicles, predictions, tracks and stops.

Pure functions over the fleet state and the plan: no I/O, no clock of
their own. The dataset clock is always passed in.
"""

from collections.abc import Iterable, Mapping
from datetime import datetime, timedelta

from app.config import ApiConfig
from app.models import PlanStop, PredictionRow
from app.schemas import (
    CardStop,
    Freshness,
    FreshnessCounts,
    PredictionView,
    RiskCounts,
    Summary,
    TrackPoint,
    VehicleView,
)
from contracts import StopEvent, TelemetryRecord


def freshness(last_seen: datetime, clock: datetime, config: ApiConfig) -> Freshness:
    """``active`` up to ``stale_after_sec`` of silence, ``stale`` up to ``drop_after_sec``.

    Both bounds are inclusive and match the prediction loop: a stale
    vehicle is still scored, an offline one is not.
    """
    silence = clock - last_seen
    if silence <= timedelta(seconds=config.stale_after_sec):
        return "active"
    if silence <= timedelta(seconds=config.drop_after_sec):
        return "stale"
    return "offline"


def has_position(record: TelemetryRecord) -> bool:
    """True if the point can be put on the map."""
    return record.location_valid and record.lat is not None and record.lon is not None


def prediction_view(row: PredictionRow, target_address: str | None) -> PredictionView:
    """The dashboard's form of a ``predictions`` row."""
    return PredictionView(
        sample_id=row.sample_id,
        t=row.t,
        target_stop_id=row.target_stop_id,
        target_address=target_address,
        target_time_begin=row.target_time_begin,
        prediction_s=row.prediction_s,
        p_late=row.p_late,
        reasons=row.reasons,
        risk_level=row.risk_level,
        degraded=row.degraded,
        degraded_reason=row.degraded_reason,
        model_version=row.model_version,
    )


def vehicle_view(
    tr_id: int,
    points: list[TelemetryRecord],
    clock: datetime,
    prediction: PredictionView | None,
    config: ApiConfig,
) -> VehicleView:
    """The map entry of a vehicle from its window of points, oldest first.

    ``points`` must not be empty. The position is the last one with a
    valid fix, even when a point without one came after it.
    """
    last = points[-1]
    fix = next((p for p in reversed(points) if has_position(p)), None)
    return VehicleView(
        tr_id=tr_id,
        unit_id=last.unit_id,
        lat=fix.lat if fix else None,
        lon=fix.lon if fix else None,
        heading_deg=fix.heading_deg if fix else None,
        speed_kmh=fix.speed_kmh if fix else None,
        last_seen=last.event_time,
        freshness=freshness(last.event_time, clock, config),
        prediction=prediction,
    )


def diff_views(
    published: Mapping[int, VehicleView], current: Mapping[int, VehicleView]
) -> tuple[list[VehicleView], list[int]]:
    """Views that differ from the published ones, by ``tr_id``, and the ids that are gone."""
    changed = [view for tr_id, view in sorted(current.items()) if published.get(tr_id) != view]
    removed = sorted(tr_id for tr_id in published if tr_id not in current)
    return changed, removed


def summarize(views: Iterable[VehicleView]) -> Summary:
    """Vehicles per risk level and per freshness."""
    risk = {"green": 0, "yellow": 0, "red": 0, "none": 0}
    fresh = {"active": 0, "stale": 0, "offline": 0}
    for view in views:
        risk[view.prediction.risk_level if view.prediction else "none"] += 1
        fresh[view.freshness] += 1
    return Summary(risk=RiskCounts(**risk), freshness=FreshnessCounts(**fresh))


def track(points: Iterable[TelemetryRecord], since: datetime, until: datetime) -> list[TrackPoint]:
    """Points with a valid position and ``event_time`` in ``[since, until]``, in the order given."""
    return [
        TrackPoint(event_time=p.event_time, lat=p.lat, lon=p.lon, speed_kmh=p.speed_kmh)
        for p in points
        if since <= p.event_time <= until and has_position(p)
    ]


def card_stops(
    stops: Iterable[PlanStop], events: Iterable[StopEvent], begin: datetime, end: datetime
) -> list[CardStop]:
    """Planned stops with ``time_plan`` in ``[begin, end]``, with the fact where one came."""
    passed = {event.stop_id: event for event in events}
    result = []
    for stop in stops:
        if not begin <= stop.time_plan <= end:
            continue
        event = passed.get(stop.stop_id)
        result.append(
            CardStop(
                stop_id=stop.stop_id,
                address=stop.address,
                lat=stop.lat,
                lon=stop.lon,
                time_plan=stop.time_plan,
                time_fact=event.time_fact if event else None,
                delay_s=event.delay_s if event else None,
            )
        )
    return result
