"""What api knows about the fleet right now, fed from the telemetry and stop_events streams.

Held in memory only and rebuilt from the same streams after a restart
(see ``app.streams``): the ingest and matcher tables are not read. The
whole state is small — tens of vehicles, a few hundred points each.
"""

import bisect
from datetime import datetime, timedelta

from contracts import StopEvent, TelemetryRecord


def _event_time(record: TelemetryRecord) -> datetime:
    return record.event_time


class FleetState:
    """Dataset clock, per-vehicle telemetry window and passed stops.

    Not thread-safe; everything runs on the api event loop.
    """

    def __init__(self, telemetry_window: timedelta) -> None:
        self._window = telemetry_window
        self.clock: datetime | None = None
        # The earliest event time seen since this state was created. Not
        # moved by pruning: it tells whether a full window has been observed.
        self.observed_since: datetime | None = None
        self._telemetry: dict[int, list[TelemetryRecord]] = {}
        self._stop_events: dict[int, dict[int, StopEvent]] = {}

    @property
    def window(self) -> timedelta:
        return self._window

    def add_telemetry(self, record: TelemetryRecord) -> None:
        """Advance the clock; keep the point if it belongs to a known vehicle.

        A record without ``tr_id`` still moves the clock: the clock is the
        dataset's time, not a vehicle's. Points arrive mostly in order; one
        that does not is inserted in its place.
        """
        t = record.event_time
        if self.clock is None or t > self.clock:
            self.clock = t
        if self.observed_since is None or t < self.observed_since:
            self.observed_since = t
        if record.tr_id is None:
            return
        points = self._telemetry.setdefault(record.tr_id, [])
        if not points or points[-1].event_time <= t:
            points.append(record)
        else:
            bisect.insort(points, record, key=_event_time)

    def add_stop_event(self, event: StopEvent) -> None:
        """Remember a passed stop; a repeated event for the same stop replaces the earlier one."""
        self._stop_events.setdefault(event.tr_id, {})[event.stop_id] = event

    def prune(self) -> None:
        """Drop telemetry older than the window before the clock; forget emptied vehicles."""
        if self.clock is None:
            return
        cutoff = self.clock - self._window
        for tr_id in list(self._telemetry):
            points = self._telemetry[tr_id]
            keep_from = bisect.bisect_left(points, cutoff, key=_event_time)
            if keep_from == len(points):
                del self._telemetry[tr_id]
            elif keep_from:
                del points[:keep_from]

    def vehicles(self) -> list[int]:
        """Vehicles with telemetry held, in id order."""
        return sorted(self._telemetry)

    def telemetry(self, tr_id: int) -> list[TelemetryRecord]:
        """The vehicle's points in time order; a copy."""
        return list(self._telemetry.get(tr_id, ()))

    def stop_events(self, tr_id: int) -> list[StopEvent]:
        """The vehicle's passed stops in order of ``time_fact``."""
        return sorted(self._stop_events.get(tr_id, {}).values(), key=lambda e: e.time_fact)

    def warming_up(self, t: datetime) -> bool:
        """True while telemetry seen so far does not reach a full window back from ``t``."""
        return self.observed_since is None or self.observed_since > t - self._window
