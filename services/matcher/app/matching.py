"""Causal point matching and conservative geometry-based arrival detection."""

import math
from bisect import bisect_left, bisect_right
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta

from app.dataset import Tick, Visit


@dataclass(frozen=True)
class Settings:
    stop_radius_m: float
    stopped_speed_kmh: float
    visit_time_tolerance_sec: float
    stale_after_sec: float
    horizon_min_sec: float
    horizon_max_sec: float

    def __post_init__(self):
        values = asdict(self)
        if any(not math.isfinite(v) or v <= 0 for v in values.values()):
            raise ValueError("Matcher thresholds must be finite and positive")
        if self.horizon_min_sec >= self.horizon_max_sec:
            raise ValueError("Invalid target horizon interval")


def distance_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    lat1, lat2 = math.radians(lat1), math.radians(lat2)
    a = (
        math.sin((lat2 - lat1) / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    )
    return 6371008.8 * 2 * math.asin(math.sqrt(min(1, a)))


@dataclass
class StopEvent:
    vehicle_id: str
    visit_id: str
    planned_at: datetime
    arrival: datetime
    departure: datetime | None
    available_at: datetime
    recovered: bool

    @property
    def delay_sec(self) -> float:
        return (self.arrival - self.planned_at).total_seconds()


class StopDetector:
    """Incremental per-vehicle detector; late ticks never rewrite emitted events.

    Visits are ordered within a vehicle-day, not presented as inferred trips.
    A pass-through is confirmed only on leaving the radius. Its earlier arrival
    must not be used before available_at when replaying historical features.
    """

    def __init__(self, visits: list[Visit], settings: Settings):
        self.settings = settings
        self.visits = defaultdict(list)
        for visit in visits:
            self.visits[(visit.vehicle_id, visit.planned_at.date())].append(visit)
        for visits_for_day in self.visits.values():
            visits_for_day.sort(key=lambda v: (v.planned_at, v.visit_id))
        self.times = {k: [v.planned_at for v in vs] for k, vs in self.visits.items()}
        self.last_tick = {}
        self.last_valid_tick = {}
        self.last_seq = defaultdict(lambda: -1)
        self.active = {}
        self.events = []
        self.rejected_late_ticks = 0

    def feed(self, tick: Tick) -> list[StopEvent]:
        previous = self.last_tick.get(tick.vehicle_id)
        if previous is not None and tick.at <= previous:
            self.rejected_late_ticks += 1
            return []
        self.last_tick[tick.vehicle_id] = tick.at
        if not tick.valid:
            return []
        previous_valid = self.last_valid_tick.get(tick.vehicle_id)
        self.last_valid_tick[tick.vehicle_id] = tick.at
        if (
            previous_valid is not None
            and (tick.at - previous_valid).total_seconds() > self.settings.stale_after_sec
        ):
            expired = self.active.pop(tick.vehicle_id, None)
            if expired and expired[4] is not None:
                # Keep an observed arrival, but do not invent a departure across a gap.
                self.last_seq[expired[0]] = expired[1]
        emitted = []
        active = self.active.get(tick.vehicle_id)
        if active:
            key, seq, visit, closest, event = active
            distance = distance_m(tick.lon, tick.lat, visit.lon, visit.lat)
            if distance > self.settings.stop_radius_m:
                if event is None:
                    event = StopEvent(
                        visit.vehicle_id,
                        visit.visit_id,
                        visit.planned_at,
                        closest.at,
                        closest.at,
                        tick.at,
                        True,
                    )
                    self.events.append(event)
                    emitted.append(event)
                else:
                    event.departure = tick.at
                    if event not in self.events:
                        self.events.append(event)
                self.last_seq[key] = seq
                del self.active[tick.vehicle_id]
            else:
                if distance < distance_m(closest.lon, closest.lat, visit.lon, visit.lat):
                    closest = tick
                if (
                    event is None
                    and tick.speed is not None
                    and (tick.speed < self.settings.stopped_speed_kmh)
                ):
                    event = self._arrival(visit, tick)
                    emitted.append(event)
                self.active[tick.vehicle_id] = (key, seq, visit, closest, event)
                return emitted
        tolerance = timedelta(seconds=self.settings.visit_time_tolerance_sec)
        candidates = []
        # Include adjacent dates for arrivals crossing midnight.
        for day in {(tick.at - tolerance).date(), tick.at.date(), (tick.at + tolerance).date()}:
            key = (tick.vehicle_id, day)
            visits = self.visits.get(key, [])
            times = self.times.get(key, [])
            left = max(self.last_seq[key] + 1, bisect_left(times, tick.at - tolerance))
            right = bisect_right(times, tick.at + tolerance)
            for seq in range(left, right):
                visit = visits[seq]
                if visit.lon is None or visit.lat is None:
                    continue
                distance = distance_m(tick.lon, tick.lat, visit.lon, visit.lat)
                if distance <= self.settings.stop_radius_m:
                    candidates.append((key, seq, visit))
        # Nearby repeated visits are ambiguous; do not invent a confident assignment.
        if len(candidates) == 1:
            key, seq, visit = candidates[0]
            event = None
            if tick.speed is not None and tick.speed < self.settings.stopped_speed_kmh:
                event = self._arrival(visit, tick)
                emitted.append(event)
            self.active[tick.vehicle_id] = (key, seq, visit, tick, event)
        return emitted

    def _arrival(self, visit: Visit, tick: Tick) -> StopEvent:
        event = StopEvent(
            visit.vehicle_id, visit.visit_id, visit.planned_at, tick.at, None, tick.at, False
        )
        self.events.append(event)
        return event

    def drain_events(self) -> list[StopEvent]:
        events, self.events = self.events, []
        return events

    def snapshot(self, vehicle: str) -> dict:
        last = self.last_tick.get(vehicle)
        oldest = (last - timedelta(days=1)).date() if last else date.min
        active = self.active.get(vehicle)
        return {
            "version": 1,
            "settings": asdict(self.settings),
            "last_tick": last.isoformat() if last else None,
            "last_valid_tick": self.last_valid_tick[vehicle].isoformat()
            if vehicle in self.last_valid_tick
            else None,
            "last_seq": {
                day.isoformat(): seq
                for (v, day), seq in self.last_seq.items()
                if v == vehicle and day >= oldest
            },
            "active": {
                "day": active[0][1].isoformat(),
                "seq": active[1],
                "visit": asdict(active[2]),
                "closest": asdict(active[3]),
                "event": asdict(active[4]) if active[4] else None,
            }
            if active
            else None,
        }

    def restore(self, vehicle: str, state: dict) -> None:
        if not state:
            return
        if state.get("version") != 1 or state.get("settings") != asdict(self.settings):
            raise ValueError("Stored matcher state uses a different version or configuration")
        for name in ("last_tick", "last_valid_tick"):
            if state[name]:
                getattr(self, name)[vehicle] = datetime.fromisoformat(state[name])
        for day, seq in state["last_seq"].items():
            self.last_seq[(vehicle, date.fromisoformat(day))] = seq
        active = state["active"]
        if active:
            visit = dict(active["visit"])
            visit["planned_at"] = datetime.fromisoformat(visit["planned_at"])
            closest = dict(active["closest"])
            closest["at"] = datetime.fromisoformat(closest["at"])
            event = dict(active["event"]) if active["event"] else None
            if event:
                for key in ("planned_at", "arrival", "departure", "available_at"):
                    if event[key]:
                        event[key] = datetime.fromisoformat(event[key])
            self.active[vehicle] = (
                (vehicle, date.fromisoformat(active["day"])),
                active["seq"],
                Visit(**visit),
                Tick(**closest),
                StopEvent(**event) if event else None,
            )
