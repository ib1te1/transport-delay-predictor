"""Features of one forecast point.

A point is a vehicle, a moment ``T`` and a target stop that the bus should reach by plan
10-15 minutes later. Features are computed only from:

* the planned timetable (known in advance, so the whole day can be used),
* telemetry with ``event_time <= T``,
* the ``cur_dev_s`` hint that comes with the point.

:func:`build_features` cuts every track at ``T`` and calls :func:`point_features`; the live
predictor calls the same function with its own buffer. Fact times from the schedule never
get here, :mod:`busdelay.data` keeps them apart.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .arrivals import detect_arrivals
from .data import LOCAL_OFFSET_S, is_synthetic
from .geo import haversine_m
from .schedule import LAYOVER_GAP_S, VehiclePlan, prepare_plan, route_ids, split_by_vehicle

# How much of the past the features look at. Long enough to see the same stop on the
# previous loop (loops are 40-90 minutes).
HISTORY_S = 150 * 60

MOVING_KMH = 5.0
# Floor for the speed in the ETA estimate, otherwise a standing bus "arrives" never.
MIN_SPEED_MS = 1.0
# An earlier visit of the same stop counts as the previous loop only if it was planned at
# least this much earlier than the target.
LOOP_MIN_S = 20 * 60
# Stops planned up to this long after T are also matched: a bus running early may have
# passed them already. Further ahead false matches (the other direction) take over.
EARLY_LOOKAHEAD_S = 2 * 60
# Real delays in the data stay under ~11 minutes, a bigger "overdue" is a missed pass.
MAX_OVERDUE_S = 15 * 60
# The bus position is looked for among the stops from this many before the last planned
# one up to the target, so that the other half of a loop is not picked.
ROUTE_LOOKBACK = 6

FEATURES = [
    # timetable
    "lead_s",
    "hour",
    "since_last_plan_s",
    "stops_ahead",
    "plan_dist_ahead_m",
    "plan_run_s",
    "layover_ahead_s",
    "new_trip_ahead",
    "last_trip_frac",
    "target_seq",
    "target_trip_frac",
    "trip_stops",
    "trip_no",
    "target_lat",
    "target_lon",
    # the hint
    "cur_dev_s",
    "layover_slack_s",
    # telemetry right before T
    "tel_age_s",
    "fixes_15m",
    "ok_share_15m",
    "speed_mean_5m",
    "speed_mean_15m",
    "speed_max_5m",
    "stopped_share_10m",
    "standing_s",
    "dist_to_target_m",
    "route_offset_m",
    "gps_dist_ahead_m",
    "gps_stops_ahead",
    "eta_dev_s",
    # delays at stops already passed, from GPS
    "gps_dev_last_s",
    "gps_dev_age_s",
    "gps_dev_mean3_s",
    "gps_dev_trend_s",
    "gps_overdue_s",
    "gps_dev_now_s",
    "gps_minus_cur_s",
    "gps_found_share",
    "dwell_last_s",
    "dwell_mean3_s",
    # the same stretch on the previous loop
    "prev_loop_dev_s",
    "prev_loop_age_s",
    "prev_loop_gain_s",
]


@dataclass
class Track:
    """Telemetry of one vehicle as numpy arrays sorted by time."""

    t: np.ndarray
    ok: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    speed: np.ndarray

    @classmethod
    def from_frame(cls, frame: pd.DataFrame) -> "Track":
        frame = frame.sort_values("t", kind="stable")
        return cls(
            t=frame["t"].to_numpy(dtype=float),
            ok=frame["ok"].to_numpy(dtype=bool),
            lat=frame["lat"].to_numpy(dtype=float),
            lon=frame["lon"].to_numpy(dtype=float),
            speed=frame["speed"].to_numpy(dtype=float),
        )

    @classmethod
    def empty(cls) -> "Track":
        nothing = np.array([], dtype=float)
        return cls(nothing, np.array([], dtype=bool), nothing, nothing, nothing)

    def __len__(self) -> int:
        return len(self.t)

    def _slice(self, start: int, end: int) -> "Track":
        return Track(
            self.t[start:end],
            self.ok[start:end],
            self.lat[start:end],
            self.lon[start:end],
            self.speed[start:end],
        )

    def upto(self, t: float) -> "Track":
        """Fixes with time <= t."""
        return self._slice(0, int(np.searchsorted(self.t, t, side="right")))

    def since(self, t: float) -> "Track":
        """Fixes with time >= t."""
        return self._slice(int(np.searchsorted(self.t, t, side="left")), len(self.t))


def point_features(
    plan: VehiclePlan, track: Track, T: float, target_stop_id: int, cur_dev_s: float
) -> dict[str, float]:
    """All features of one point as a dict with the keys of :data:`FEATURES`.

    ``track`` must not contain fixes after ``T``. Missing values are NaN, CatBoost handles
    them.
    """
    if len(track) and track.t[-1] > T:
        raise ValueError("track has fixes after T, cut it with Track.upto(T) first")
    target = plan.index_of(target_stop_id)
    last = plan.last_planned(T)
    if target <= last:
        raise ValueError(f"target stop {target_stop_id} is planned before T")

    f = dict.fromkeys(FEATURES, np.nan)
    _timetable(f, plan, T, target, last)

    f["cur_dev_s"] = cur_dev_s
    if f["layover_ahead_s"] > 0 and not np.isnan(cur_dev_s):
        f["layover_slack_s"] = f["layover_ahead_s"] - max(cur_dev_s, 0.0)

    recent = track.since(T - HISTORY_S)
    position = _route_position(plan, recent, last, target)
    _motion(f, recent, plan, T, target, position)
    _arrival_history(f, recent, plan, T, target, position)
    if not np.isnan(f["gps_dev_last_s"]) and not np.isnan(cur_dev_s):
        f["gps_minus_cur_s"] = f["gps_dev_last_s"] - cur_dev_s
    return f


def estimate_cur_dev(plan: VehiclePlan, track: Track, T: float) -> float:
    """What ``cur_dev_s`` would be, from GPS only. Used on the live stream.

    ``cur_dev_s`` is the delay at the last stop planned at or before ``T``. If the bus has
    been seen there, that is the answer. If the GPS says the bus is already at or past that
    stop but the pass was not caught (often at a terminal), the last delay seen is used. If
    the bus is still behind, it is at least ``T - planned time`` late and probably no less
    late than at the last stop seen. NaN before the first stop of the day or with no
    telemetry at all.
    """
    if len(track) and track.t[-1] > T:
        raise ValueError("track has fixes after T, cut it with Track.upto(T) first")
    last = plan.last_planned(T)
    if last < 0:
        return np.nan
    recent = track.since(T - HISTORY_S)
    position = _route_position(plan, recent, last, min(last + 1, len(plan) - 1))
    if position is None:
        return np.nan
    lo = int(np.searchsorted(plan.t_plan, T - HISTORY_S, side="left"))
    if lo > last:
        # the last planned stop is older than the history, a long break in the plan
        return np.nan
    good = recent.ok
    arrived, _, _ = detect_arrivals(
        recent.t[good],
        recent.lat[good],
        recent.lon[good],
        plan.t_plan[lo : last + 1],
        plan.lat[lo : last + 1],
        plan.lon[lo : last + 1],
        departs=plan.seq[lo : last + 1] == 0,
    )
    arrived[np.arange(lo, last + 1) > position[0]] = np.nan
    if not np.isnan(arrived[-1]):
        return float(arrived[-1] - plan.t_plan[last])

    seen = np.flatnonzero(~np.isnan(arrived))
    last_seen = arrived[seen[-1]] - plan.t_plan[lo + seen[-1]] if seen.size else 0.0
    if position[0] >= last:
        return float(last_seen)
    overdue = min(T - plan.t_plan[last], MAX_OVERDUE_S)
    return float(max(last_seen, overdue))


def _timetable(f: dict, plan: VehiclePlan, T: float, target: int, last: int) -> None:
    t_target = plan.t_plan[target]
    f["lead_s"] = t_target - T
    f["hour"] = ((T + LOCAL_OFFSET_S) % 86400) / 3600
    f["target_seq"] = plan.seq[target]
    f["target_trip_frac"] = plan.seq[target] / max(plan.trip_len[target] - 1, 1)
    f["trip_stops"] = plan.trip_len[target]
    # where and in which trip of the day: lets the model learn a delay profile of a stop
    f["trip_no"] = plan.trip[target]
    f["target_lat"] = plan.lat[target]
    f["target_lon"] = plan.lon[target]

    if last >= 0:
        f["since_last_plan_s"] = T - plan.t_plan[last]
        f["stops_ahead"] = target - last
        f["plan_dist_ahead_m"] = plan.cum_m[target] - plan.cum_m[last]
        f["plan_run_s"] = t_target - plan.t_plan[last]
        f["last_trip_frac"] = plan.seq[last] / max(plan.trip_len[last] - 1, 1)
        f["new_trip_ahead"] = float(plan.trip[target] != plan.trip[last])
        gaps = plan.gap_s[last + 1 : target + 1]
    else:
        # the vehicle's day has not started yet
        f["stops_ahead"] = target + 1
        f["new_trip_ahead"] = 1.0
        gaps = plan.gap_s[1 : target + 1]
    f["layover_ahead_s"] = float(gaps[gaps >= LAYOVER_GAP_S].sum())


def _route_position(plan: VehiclePlan, track: Track, last: int, target: int):
    """Planned stop nearest to the last good fix: ``(index, fix index, distance)`` or None."""
    good = np.flatnonzero(track.ok)
    if good.size == 0:
        return None
    i = good[-1]
    lo = max(last - ROUTE_LOOKBACK, 0)
    d = haversine_m(
        plan.lat[lo : target + 1], plan.lon[lo : target + 1], track.lat[i], track.lon[i]
    )
    k = int(np.argmin(d))
    return lo + k, i, float(d[k])


def _motion(f: dict, track: Track, plan: VehiclePlan, T: float, target: int, position) -> None:
    window15 = track.t >= T - 15 * 60
    f["fixes_15m"] = float(window15.sum())
    if window15.any():
        f["ok_share_15m"] = float(track.ok[window15].mean())

    has_speed = track.ok & ~np.isnan(track.speed)
    for minutes in (5, 15):
        sel = has_speed & (track.t >= T - minutes * 60)
        if sel.any():
            f[f"speed_mean_{minutes}m"] = float(track.speed[sel].mean())
            if minutes == 5:
                f["speed_max_5m"] = float(track.speed[sel].max())
    sel = has_speed & (track.t >= T - 10 * 60)
    if sel.any():
        f["stopped_share_10m"] = float((track.speed[sel] < MOVING_KMH).mean())
    moving = np.flatnonzero(has_speed & (track.speed >= MOVING_KMH))
    if moving.size:
        f["standing_s"] = T - track.t[moving[-1]]
    elif has_speed.any():
        f["standing_s"] = float(HISTORY_S)

    if position is None:
        return
    k, i, offset = position
    f["tel_age_s"] = T - track.t[i]
    f["dist_to_target_m"] = float(
        haversine_m(track.lat[i], track.lon[i], plan.lat[target], plan.lon[target])
    )
    f["route_offset_m"] = offset
    f["gps_dist_ahead_m"] = plan.cum_m[target] - plan.cum_m[k]
    f["gps_stops_ahead"] = target - k

    if not np.isnan(f["speed_mean_15m"]):
        speed_ms = max(f["speed_mean_15m"] / 3.6, MIN_SPEED_MS)
        eta = T + f["gps_dist_ahead_m"] / speed_ms
        f["eta_dev_s"] = float(np.clip(eta - plan.t_plan[target], -1800, 1800))


def _arrival_history(
    f: dict, track: Track, plan: VehiclePlan, T: float, target: int, position
) -> None:
    lo = int(np.searchsorted(plan.t_plan, T - HISTORY_S, side="left"))
    hi = min(target, int(np.searchsorted(plan.t_plan, T + EARLY_LOOKAHEAD_S, side="right")))
    if hi <= lo or position is None:
        return
    good = track.ok
    arrived, dwell, _ = detect_arrivals(
        track.t[good],
        track.lat[good],
        track.lon[good],
        plan.t_plan[lo:hi],
        plan.lat[lo:hi],
        plan.lon[lo:hi],
        departs=plan.seq[lo:hi] == 0,
    )
    # A stop ahead of where the bus is now cannot have been reached yet, such a match is
    # a pass of the other direction or of another part of the loop.
    arrived[np.arange(lo, hi) > position[0]] = np.nan
    dev = arrived - plan.t_plan[lo:hi]
    found = ~np.isnan(arrived)

    recent_plan = (plan.t_plan[lo:hi] >= T - 30 * 60) & (plan.t_plan[lo:hi] <= T)
    if recent_plan.any():
        f["gps_found_share"] = float(found[recent_plan].mean())

    seen = np.flatnonzero(found)
    if seen.size == 0:
        return
    k = seen[-1]
    f["gps_dev_last_s"] = dev[k]
    f["gps_dev_age_s"] = T - arrived[k]
    f["gps_dev_mean3_s"] = float(dev[seen[-3:]].mean())
    f["gps_dev_trend_s"] = dev[k] - dev[seen[max(len(seen) - 6, 0)]]
    f["dwell_last_s"] = dwell[k]
    f["dwell_mean3_s"] = float(dwell[seen[-3:]].mean())

    # The next stop after the last one we saw. If its planned time is already behind, the
    # bus is at least that late (like the interpolation in the design spec).
    nxt = lo + k + 1
    if nxt < len(plan):
        f["gps_overdue_s"] = T - plan.t_plan[nxt]
        f["gps_dev_now_s"] = max(dev[k], f["gps_overdue_s"])

    # Previous loop: the latest earlier visit of the target stop that we saw, and how much
    # delay the bus gained on the way to it from the stop matching our last seen one.
    keys = plan.key[lo:hi]
    earlier = np.flatnonzero(
        found
        & (keys == plan.key[target])
        & (plan.t_plan[lo:hi] <= plan.t_plan[target] - LOOP_MIN_S)
    )
    if earlier.size == 0:
        return
    e = earlier[-1]
    f["prev_loop_dev_s"] = dev[e]
    f["prev_loop_age_s"] = T - arrived[e]
    refs = np.flatnonzero(found[:e] & (keys[:e] == keys[k]))
    if refs.size:
        f["prev_loop_gain_s"] = dev[e] - dev[refs[-1]]


def read_feature_table(directory, parts) -> pd.DataFrame:
    """Load ``<part>.parquet`` files written by ``python -m busdelay features``."""
    frames = [pd.read_parquet(Path(directory) / f"{part}.parquet") for part in parts]
    return pd.concat(frames, ignore_index=True)


def build_features(
    points: pd.DataFrame, plan: pd.DataFrame, telemetry: pd.DataFrame, hint: str = "given"
) -> pd.DataFrame:
    """Features for a batch of points, one row per point in the same order.

    Args:
        points: frame from :func:`busdelay.data.read_points`.
        plan: planned stops from :func:`busdelay.data.read_plan`. A frame with fact times is
            refused.
        telemetry: cleaned telemetry from :func:`busdelay.data.clean_telemetry`.
        hint: ``"given"`` takes ``cur_dev_s`` from the points, ``"gps"`` replaces it with
            :func:`estimate_cur_dev` - what the live stream will have.

    Returns:
        ``sample_id, tr_id, part, T, target, synthetic, route`` followed by
        :data:`FEATURES`. ``route`` groups a real vehicle with its synthetic copies.
    """
    if any("fact" in column for column in plan.columns):
        raise ValueError("plan has fact columns, features must be built from the plan only")
    if hint not in ("given", "gps"):
        raise ValueError(f"hint must be 'given' or 'gps', got {hint!r}")
    plans = split_by_vehicle(prepare_plan(plan))
    tracks = {int(tr_id): Track.from_frame(rows) for tr_id, rows in telemetry.groupby("tr_id")}

    rows = []
    for p in points.itertuples(index=False):
        track = tracks.get(p.tr_id, Track.empty()).upto(p.T)
        vehicle = plans[p.tr_id]
        cur_dev = p.cur_dev_s if hint == "given" else estimate_cur_dev(vehicle, track, p.T)
        rows.append(point_features(vehicle, track, p.T, p.target_stop_id, cur_dev))

    meta = points[["sample_id", "tr_id", "part", "T", "target"]].reset_index(drop=True)
    meta["synthetic"] = is_synthetic(meta["tr_id"])
    meta["route"] = meta["tr_id"].map(route_ids(plan))
    return pd.concat([meta, pd.DataFrame(rows, columns=FEATURES)], axis=1)
