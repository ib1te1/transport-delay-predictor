"""Load reference data for the demo period: planned stops and terminal ids.

Run as ``python -m app.seed``; compose runs it once after migrations. The
fact column of the schedule is never read: a fact inside the system would
leak into stop detection and prediction requests.

A missing dataset is not an error: the system starts empty, and a
developer without the dataset can still run everything else.
"""

import csv
import logging
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import psycopg

from app.config import SeedConfig
from app.models import PlanStop, Vehicle
from common.config import ServiceSettings, load_section
from common.db import connect, insert_models

log = logging.getLogger(__name__)

# (schedule, traffic) per dataset part, relative to the dataset root.
PERIOD_FILES = {
    "train": ("train/schedule.csv", "train/traffic.csv"),
    "test": ("test/schedule.csv", "test/traffic.csv"),
    "validate": ("validate/schedule_plan.csv", "validate/traffic.csv"),
}

_POINT = re.compile(r"^\s*POINT\s*\(\s*(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s*\)\s*$")


def parse_point(value: str) -> tuple[float, float]:
    """``POINT (lon lat)`` as a ``(lon, lat)`` pair."""
    match = _POINT.match(value)
    if match is None:
        raise ValueError(f"not a WKT point: {value!r}")
    return float(match[1]), float(match[2])


def parse_timestamp(value: str, tz: ZoneInfo) -> datetime:
    """Naive dataset timestamp, possibly with nanoseconds, as an aware datetime."""
    text = value.strip()
    head, dot, fraction = text.partition(".")
    if dot:
        text = f"{head}.{fraction[:6]}"
    return datetime.fromisoformat(text).replace(tzinfo=tz)


def read_plan_stops(path: Path, tz: ZoneInfo) -> list[PlanStop]:
    """Planned arrivals from a schedule file; only plan columns are read."""
    stops = []
    with path.open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            lon, lat = parse_point(row["geom"])
            stops.append(
                PlanStop(
                    stop_id=int(row["tt_action_item_id"]),
                    tr_id=int(row["tr_id"]),
                    time_plan=parse_timestamp(row["time_begin"], tz),
                    lat=lat,
                    lon=lon,
                    address=row.get("building_address") or None,
                )
            )
    return stops


def read_vehicles(path: Path) -> list[Vehicle]:
    """Distinct terminal → vehicle pairs from a telemetry file."""
    pairs: dict[int, int] = {}
    with path.open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            if not row["unit_id"] or not row["tr_id"]:
                continue
            unit_id, tr_id = int(row["unit_id"]), int(row["tr_id"])
            known = pairs.setdefault(unit_id, tr_id)
            if known != tr_id:
                log.warning(
                    "unit %s maps to vehicles %s and %s; keeping %s", unit_id, known, tr_id, known
                )
    return [Vehicle(unit_id=u, tr_id=t) for u, t in pairs.items()]


def run_seed(conn: psycopg.Connection, data_dir: Path, config: SeedConfig) -> tuple[int, int]:
    """Upsert the period's reference data; returns (stops, vehicles). Does not commit."""
    schedule_rel, traffic_rel = PERIOD_FILES[config.period]
    schedule, traffic = data_dir / schedule_rel, data_dir / traffic_rel
    missing = [str(p) for p in (schedule, traffic) if not p.is_file()]
    if missing:
        log.warning("dataset files not found, nothing loaded: %s", ", ".join(missing))
        return 0, 0

    stops = read_plan_stops(schedule, ZoneInfo(config.source_timezone))
    vehicles = read_vehicles(traffic)
    insert_models(
        conn,
        "stops_plan",
        stops,
        on_conflict="ON CONFLICT (stop_id) DO UPDATE SET tr_id = EXCLUDED.tr_id,"
        " time_plan = EXCLUDED.time_plan, lat = EXCLUDED.lat, lon = EXCLUDED.lon,"
        " address = EXCLUDED.address",
    )
    insert_models(
        conn,
        "vehicles",
        vehicles,
        on_conflict="ON CONFLICT (unit_id) DO UPDATE SET tr_id = EXCLUDED.tr_id",
    )
    return len(stops), len(vehicles)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    settings = ServiceSettings()
    config = load_section(settings.config_path, "seed", SeedConfig)
    with connect(settings.database_url) as conn:
        stops, vehicles = run_seed(conn, settings.data_dir, config)
        conn.commit()
    log.info("seed %s: %d planned stops, %d terminals", config.period, stops, vehicles)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
