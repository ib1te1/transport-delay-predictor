"""Per-vehicle view of the timetable.

The schedule has no route or trip ids, only vehicles and their planned stops. Trips are cut
at long pauses in the plan: when nothing is planned for ``LAYOVER_GAP_S`` or longer, the bus
stands at a terminal between two trips.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .geo import haversine_m

# Inside a trip stops are planned 1-3 minutes apart, 4-6 minutes is rare.
# 7 minutes and more is almost always a terminal.
LAYOVER_GAP_S = 7 * 60

# The task: predict the delay at the first stop planned in (T + 10 min, T + 15 min].
LEAD_MIN_S = 10 * 60
LEAD_MAX_S = 15 * 60


def prepare_plan(plan: pd.DataFrame) -> pd.DataFrame:
    """Sort the stops of every vehicle and add the trip structure.

    Added columns:

    * ``pos`` - index of the stop in the vehicle's day
    * ``gap_s`` - planned time since the previous stop, NaN for the first one
    * ``trip`` - trip number within the day, ``seq`` - stop number within the trip,
      ``trip_len`` - number of stops in the trip
    * ``step_m`` and ``cum_m`` - straight-line distance from the previous stop and its
      running sum over the day
    * ``key`` - rounded coordinates, the same physical stop gets the same key on every loop
    """
    df = plan.sort_values(["tr_id", "t_plan", "stop_id"], kind="stable").reset_index(drop=True)
    vehicle = df.groupby("tr_id", sort=False)
    df["pos"] = vehicle.cumcount()
    df["gap_s"] = vehicle["t_plan"].diff()

    new_trip = df["gap_s"].isna() | (df["gap_s"] >= LAYOVER_GAP_S)
    df["trip"] = new_trip.astype(int).groupby(df["tr_id"]).cumsum() - 1
    trip = df.groupby(["tr_id", "trip"], sort=False)
    df["seq"] = trip.cumcount()
    df["trip_len"] = trip["stop_id"].transform("size")

    step = haversine_m(vehicle["lat"].shift(), vehicle["lon"].shift(), df["lat"], df["lon"])
    df["step_m"] = np.nan_to_num(step, nan=0.0)
    df["cum_m"] = df.groupby("tr_id", sort=False)["step_m"].cumsum()
    df["key"] = df["lon"].round(5).astype(str) + "," + df["lat"].round(5).astype(str)
    return df


@dataclass
class VehiclePlan:
    """Timetable of one vehicle as numpy arrays, sorted by planned time."""

    tr_id: int
    stop_id: np.ndarray
    t_plan: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    gap_s: np.ndarray
    trip: np.ndarray
    seq: np.ndarray
    trip_len: np.ndarray
    cum_m: np.ndarray
    key: np.ndarray

    @classmethod
    def from_frame(cls, frame: pd.DataFrame) -> "VehiclePlan":
        """Build from the rows of one vehicle in a frame made by :func:`prepare_plan`."""
        ids = frame["tr_id"].unique()
        if len(ids) != 1:
            raise ValueError(f"expected stops of one vehicle, got {len(ids)}")
        frame = frame.sort_values("pos")
        return cls(
            tr_id=int(ids[0]),
            stop_id=frame["stop_id"].to_numpy(),
            t_plan=frame["t_plan"].to_numpy(dtype=float),
            lat=frame["lat"].to_numpy(dtype=float),
            lon=frame["lon"].to_numpy(dtype=float),
            gap_s=frame["gap_s"].to_numpy(dtype=float),
            trip=frame["trip"].to_numpy(),
            seq=frame["seq"].to_numpy(),
            trip_len=frame["trip_len"].to_numpy(),
            cum_m=frame["cum_m"].to_numpy(dtype=float),
            key=frame["key"].to_numpy(),
        )

    def __len__(self) -> int:
        return len(self.stop_id)

    def index_of(self, stop_id: int) -> int:
        """Position of a stop in the day. Raises ``KeyError`` for a stop of another vehicle."""
        hits = np.flatnonzero(self.stop_id == stop_id)
        if hits.size == 0:
            raise KeyError(f"stop {stop_id} is not in the plan of vehicle {self.tr_id}")
        return int(hits[0])

    def last_planned(self, t: float) -> int:
        """Index of the last stop planned at or before ``t``, -1 before the first one."""
        return int(np.searchsorted(self.t_plan, t, side="right")) - 1

    def pick_target(self, t: float) -> int:
        """First stop planned in ``(t + 10 min, t + 15 min]``, -1 if there is none.

        This is how the organisers chose ``target_stop_id``; ``bench`` uses it to pick the
        stop to forecast for.
        """
        first = int(np.searchsorted(self.t_plan, t + LEAD_MIN_S, side="right"))
        if first < len(self) and self.t_plan[first] <= t + LEAD_MAX_S:
            return first
        return -1


def route_ids(plan: pd.DataFrame, min_shared: float = 0.5) -> dict[int, int]:
    """Vehicles driving the same stops get the same route id (the smallest tr_id among them).

    In train every real vehicle has two synthetic copies: the same stops, the day shifted by
    a few minutes, almost the same delays. For an honest check they have to be held out
    together with the real vehicle.
    """
    stops = {
        int(tr_id): set(rows["lon"].round(5).astype(str) + "," + rows["lat"].round(5).astype(str))
        for tr_id, rows in plan.groupby("tr_id")
    }
    route = {}
    for tr_id in sorted(stops):
        for other in sorted(route):
            if route[other] != other:
                continue
            shared = len(stops[tr_id] & stops[other]) / max(len(stops[tr_id]), 1)
            if shared >= min_shared:
                route[tr_id] = other
                break
        else:
            route[tr_id] = tr_id
    return route


def split_by_vehicle(plan: pd.DataFrame) -> dict[int, VehiclePlan]:
    """Prepared plan -> ``{tr_id: VehiclePlan}``."""
    return {
        int(tr_id): VehiclePlan.from_frame(rows)
        for tr_id, rows in plan.groupby("tr_id", sort=False)
    }
