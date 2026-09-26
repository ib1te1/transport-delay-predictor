"""Reading the organisers' dataset.

Expected layout, the archive unpacked as is::

    train/traffic.csv     train/schedule.csv          labels/labels_train.csv
    test/traffic.csv      test/schedule.csv           labels/labels_test.csv
    validate/traffic.csv  validate/schedule_plan.csv  validate/points.csv

Timestamps in the files are UTC without a zone (the suffix of ``sample_id`` is ``T`` as unix
time). Inside the package time is a float number of seconds since the epoch.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .geo import parse_points

PARTS = ("train", "test", "validate")

FILES = {
    "train": ("train/traffic.csv", "train/schedule.csv", "labels/labels_train.csv"),
    "test": ("test/traffic.csv", "test/schedule.csv", "labels/labels_test.csv"),
    "validate": ("validate/traffic.csv", "validate/schedule_plan.csv", "validate/points.csv"),
}

# Train has extra synthetic vehicles "for volume", their ids start at 9000000.
# Test and validate have real vehicles only.
SYNTHETIC_FROM = 9_000_000

# Moscow with a margin. Invalid fixes come as zeros.
BBOX_LAT = (55.0, 56.5)
BBOX_LON = (36.5, 38.5)
MAX_SPEED_KMH = 150.0

# Europe/Moscow, no daylight saving time.
LOCAL_OFFSET_S = 3 * 3600

EPOCH = pd.Timestamp("1970-01-01")


def to_seconds(values) -> np.ndarray:
    """Parse timestamps (strings or datetimes) into seconds since the epoch."""
    stamps = pd.to_datetime(pd.Series(values), format="ISO8601")
    return ((stamps - EPOCH) / pd.Timedelta(seconds=1)).to_numpy(dtype=float)


def is_synthetic(tr_id):
    """True for vehicles the organisers generated to pad the train part."""
    return np.asarray(tr_id) >= SYNTHETIC_FROM


@dataclass
class Part:
    """One part of the dataset (train, test or validate) with our column names.

    ``plan`` never contains fact times. For train and test the facts are kept separately in
    ``facts`` and are only used to check the arrival detector, never for features.
    """

    name: str
    telemetry: pd.DataFrame
    plan: pd.DataFrame
    points: pd.DataFrame
    facts: pd.DataFrame | None = None


def load_part(root: Path | str, name: str) -> Part:
    """Read one part from the unpacked dataset directory."""
    if name not in FILES:
        raise ValueError(f"unknown part {name!r}, expected one of: {', '.join(PARTS)}")
    traffic_file, schedule_file, points_file = (Path(root) / f for f in FILES[name])

    schedule = pd.read_csv(schedule_file)
    facts = read_facts(schedule) if "time_fact_begin" in schedule.columns else None
    return Part(
        name=name,
        telemetry=read_telemetry(traffic_file),
        plan=read_plan(schedule),
        points=read_points(points_file, part=name),
        facts=facts,
    )


def read_telemetry(path: Path | str) -> pd.DataFrame:
    """Load traffic.csv and clean it, see :func:`clean_telemetry`."""
    raw = pd.read_csv(
        path, usecols=["tr_id", "event_time", "location_valid", "lat", "lon", "speed"]
    )
    return clean_telemetry(raw)


def clean_telemetry(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalise raw telemetry rows.

    Returns columns ``tr_id, t, ok, lat, lon, speed`` sorted by vehicle and time. ``ok`` marks
    fixes with usable coordinates. Bad rows are kept with empty coordinates because the share
    of bad fixes is used as a feature.
    """
    # copies: with copy-on-write in pandas 3 to_numpy() can return a read-only view
    lat = pd.to_numeric(raw["lat"], errors="coerce").to_numpy(dtype=float, copy=True)
    lon = pd.to_numeric(raw["lon"], errors="coerce").to_numpy(dtype=float, copy=True)
    speed = pd.to_numeric(raw["speed"], errors="coerce").to_numpy(dtype=float, copy=True)
    valid = raw["location_valid"].astype(str).str.lower().eq("true").to_numpy()

    ok = (
        valid
        & (lat >= BBOX_LAT[0])
        & (lat <= BBOX_LAT[1])
        & (lon >= BBOX_LON[0])
        & (lon <= BBOX_LON[1])
    )
    lat[~ok] = np.nan
    lon[~ok] = np.nan
    speed[(speed < 0) | (speed > MAX_SPEED_KMH)] = np.nan

    df = pd.DataFrame(
        {
            "tr_id": raw["tr_id"].astype("int64").to_numpy(),
            "t": to_seconds(raw["event_time"]),
            "ok": ok,
            "lat": lat,
            "lon": lon,
            "speed": speed,
        }
    )
    df = df.drop_duplicates(subset=["tr_id", "t", "lat", "lon", "speed"])
    return df.sort_values(["tr_id", "t"], kind="stable").reset_index(drop=True)


def read_plan(schedule: pd.DataFrame) -> pd.DataFrame:
    """Planned stops: ``stop_id, tr_id, t_plan, lat, lon, address``. Fact columns are dropped."""
    coords = parse_points(schedule["geom"])
    plan = pd.DataFrame(
        {
            "stop_id": schedule["tt_action_item_id"].astype("int64").to_numpy(),
            "tr_id": schedule["tr_id"].astype("int64").to_numpy(),
            "t_plan": to_seconds(schedule["time_begin"]),
            "lat": coords["lat"].to_numpy(),
            "lon": coords["lon"].to_numpy(),
            "address": schedule["building_address"].to_numpy(),
        }
    )
    repeated = plan["stop_id"].duplicated()
    if repeated.any():
        raise ValueError(f"stop id {plan.loc[repeated, 'stop_id'].iloc[0]} appears twice")
    return plan


def read_facts(schedule: pd.DataFrame) -> pd.DataFrame:
    """Fact arrival times from train/test schedules: ``stop_id, t_fact, manual_fill``."""
    return pd.DataFrame(
        {
            "stop_id": schedule["tt_action_item_id"].astype("int64").to_numpy(),
            "t_fact": to_seconds(schedule["time_fact_begin"]),
            "manual_fill": schedule["manual_fill"].astype(str).str.lower().eq("true").to_numpy(),
        }
    )


def read_points(path: Path | str, part: str) -> pd.DataFrame:
    """Forecast points. ``target`` is the delay in seconds, NaN for validate."""
    raw = pd.read_csv(path)
    target = raw["target_delay_s"] if "target_delay_s" in raw.columns else np.nan
    return pd.DataFrame(
        {
            "sample_id": raw["sample_id"].astype(str),
            "tr_id": raw["tr_id"].astype("int64"),
            "T": to_seconds(raw["T"]),
            "target_stop_id": raw["target_stop_id"].astype("int64"),
            "t_target": to_seconds(raw["target_time_begin"]),
            "cur_dev_s": pd.to_numeric(raw["cur_dev_s"], errors="coerce"),
            "target": pd.to_numeric(target, errors="coerce"),
            "part": part,
        }
    )
