"""What one run of the prediction loop knows, as the dashboard reads it.

A ``LiveState`` is built by each run of the loop next to its
``FleetState`` and dropped with it: after a restart the current
predictions are reloaded from the ``predictions`` table, the rest from
the streams. Not thread-safe; everything runs on the api event loop.
"""

from datetime import datetime

from app.config import ApiConfig
from app.models import PlanStop, PredictionRow
from app.planner import PlanIndex
from app.schemas import PredictionView, VehicleView
from app.state import FleetState
from app.views import freshness, prediction_view, vehicle_view


class CurrentPredictions:
    """The latest prediction of each vehicle, while it still applies."""

    def __init__(self) -> None:
        self._rows: dict[int, PredictionRow] = {}

    def put(self, row: PredictionRow) -> None:
        """Keep ``row`` as the vehicle's prediction unless a newer one is already held."""
        held = self._rows.get(row.tr_id)
        if held is None or row.t >= held.t:
            self._rows[row.tr_id] = row

    def get(self, tr_id: int) -> PredictionRow | None:
        return self._rows.get(tr_id)

    def discard(self, tr_id: int) -> None:
        self._rows.pop(tr_id, None)

    def vehicles(self) -> list[int]:
        """Vehicles with a prediction held, in id order."""
        return sorted(self._rows)


class LiveState:
    """The fleet, the plan and the current predictions of one loop run."""

    def __init__(self, fleet: FleetState, plan: PlanIndex) -> None:
        self.fleet = fleet
        self.plan = plan
        self.current = CurrentPredictions()
        self._stops: dict[int, PlanStop] = {
            stop.stop_id: stop for stops in plan.values() for stop in stops
        }

    def address(self, stop_id: int) -> str | None:
        """The planned stop's address, ``None`` if unknown."""
        stop = self._stops.get(stop_id)
        return stop.address if stop else None

    def prediction_view(self, row: PredictionRow) -> PredictionView:
        """The dashboard's form of ``row``, with the target's address."""
        return prediction_view(row, self.address(row.target_stop_id))

    def views(self, config: ApiConfig) -> dict[int, VehicleView]:
        """Every vehicle in the telemetry window as the map shows it now, by ``tr_id``.

        First drops telemetry older than the window, so a vehicle whose
        last point fell out of it is gone from the result, and ends the
        current predictions that no longer apply (see ``expire``).
        """
        self.fleet.prune()
        clock = self.fleet.clock
        if clock is None:
            return {}
        self.expire(clock, config)
        views = {}
        for tr_id in self.fleet.vehicles():
            row = self.current.get(tr_id)
            prediction = self.prediction_view(row) if row else None
            views[tr_id] = vehicle_view(
                tr_id, self.fleet.telemetry(tr_id), clock, prediction, config
            )
        return views

    def expire(self, clock: datetime, config: ApiConfig) -> None:
        """End predictions whose target stop was passed or whose vehicle is offline or gone.

        An ended prediction does not come back: the vehicle shows none
        until a later tick finds it a new target.
        """
        for tr_id in self.current.vehicles():
            row = self.current.get(tr_id)
            points = self.fleet.telemetry(tr_id)
            passed = any(e.stop_id == row.target_stop_id for e in self.fleet.stop_events(tr_id))
            offline = not points or freshness(points[-1].event_time, clock, config) == "offline"
            if passed or offline:
                self.current.discard(tr_id)
