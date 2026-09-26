"""Stop arrivals recovered from the GPS track.

How the organisers' fact times look in train and test (stops not filled manually):

* ordinary stop - the fix closest to the stop: median offset 0 s, the bus is ~13 m away;
* first stop of a trip - the bus stands at the terminal, the fact is when it leaves. The
  last fix of the stay that is still as close as the closest one (+5 m) gives median 0 s.

Only the fixes passed in are used. The feature code passes fixes up to ``T``, so an arrival
is known at the same moment it would be known on the live stream.
"""

import numpy as np

from .geo import haversine_m

STOP_RADIUS_M = 80.0

# Where a pass is searched for, relative to the planned time. Real delays in the data are
# roughly from -6 to +11 minutes.
SEARCH_BEFORE_S = 7 * 60
SEARCH_AFTER_S = 15 * 60

# For a departure: fixes this much farther than the closest one still count as standing.
DEPART_TOLERANCE_M = 5.0


def detect_arrivals(
    fix_t,
    fix_lat,
    fix_lon,
    stop_t,
    stop_lat,
    stop_lon,
    departs=None,
    radius_m=STOP_RADIUS_M,
):
    """Match planned stops of one vehicle to passes of its GPS track.

    Args:
        fix_t, fix_lat, fix_lon: good fixes sorted by time.
        stop_t, stop_lat, stop_lon: planned stops sorted by planned time.
        departs: optional bool array, True for stops where the departure is wanted (the
            first stop of a trip).
        radius_m: a fix closer than this is "at the stop".

    Returns:
        Three arrays with one value per stop: arrival (or departure) time, time spent inside
        the radius (dwell) and distance of the closest fix. NaN where no pass was found.

    A pass is a run of consecutive fixes inside the radius. The expected time of a stop is
    its planned time plus the last detected delay. If there are several passes (loops, the
    stop of the opposite direction across the street), the one closest to the expected time
    wins. Arrivals never go back in time.
    """
    n = len(stop_t)
    arrival = np.full(n, np.nan)
    dwell = np.full(n, np.nan)
    closest = np.full(n, np.nan)
    if len(fix_t) == 0:
        return arrival, dwell, closest
    if departs is None:
        departs = np.zeros(n, dtype=bool)

    not_before = -np.inf
    delay = 0.0
    for i in range(n):
        lo = np.searchsorted(fix_t, max(stop_t[i] - SEARCH_BEFORE_S, not_before), side="left")
        hi = np.searchsorted(fix_t, stop_t[i] + SEARCH_AFTER_S, side="right")
        if hi <= lo:
            continue
        dist = haversine_m(fix_lat[lo:hi], fix_lon[lo:hi], stop_lat[i], stop_lon[i])

        expected = stop_t[i] + delay
        best = None
        for start, end in _runs(dist <= radius_m):
            j = start + int(np.argmin(dist[start:end]))
            if departs[i]:
                near = np.flatnonzero(dist[start:end] <= dist[j] + DEPART_TOLERANCE_M)
                j = start + int(near[-1])
            miss = abs(fix_t[lo + j] - expected)
            if best is None or miss < best[0]:
                best = (miss, j, start, end)
        if best is None:
            continue

        _, j, start, end = best
        arrival[i] = fix_t[lo + j]
        dwell[i] = fix_t[lo + end - 1] - fix_t[lo + start]
        closest[i] = dist[start:end].min()
        not_before = arrival[i]
        delay = arrival[i] - stop_t[i]
    return arrival, dwell, closest


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """``(start, end)`` of every run of True values, end exclusive."""
    if not mask.any():
        return []
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True))
