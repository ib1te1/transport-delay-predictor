"""Geometry helpers. Coordinates are in degrees, distances in metres."""

import numpy as np
import pandas as pd

EARTH_RADIUS_M = 6_371_000.0


def haversine_m(lat1, lon1, lat2, lon2):
    """Great-circle distance between two points or two arrays of points."""
    lat1, lon1, lat2, lon2 = (
        np.radians(np.asarray(value, dtype=float)) for value in (lat1, lon1, lat2, lon2)
    )
    a = (
        np.sin((lat2 - lat1) / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    )
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a))


def parse_points(geom: pd.Series) -> pd.DataFrame:
    """Split WKT strings like ``POINT (37.43 55.80)`` into ``lon`` and ``lat`` columns."""
    coords = geom.str.extract(r"POINT\s*\(\s*([-\d.eE]+)\s+([-\d.eE]+)\s*\)")
    broken = coords.isna().any(axis=1)
    if broken.any():
        raise ValueError(f"cannot parse stop geometry: {geom[broken].iloc[0]!r}")
    return pd.DataFrame(
        {"lon": coords[0].astype(float), "lat": coords[1].astype(float)}, index=geom.index
    )
