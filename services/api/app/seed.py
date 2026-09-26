"""Load reference data for the demo period: planned stops and terminal ids.

Run as ``python -m app.seed``; compose runs it once after migrations. The
fact column of the schedule is never read: a fact inside the system would
leak into stop detection and prediction requests.

A missing dataset is not an error: the system starts empty, and a
developer without the dataset can still run everything else. A malformed
row inside a present dataset is skipped rather than failing the whole
run: one bad row in a large export must not take the loader down.
"""

import csv
import logging
import re
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from zoneinfo import ZoneInfo

import psycopg

from app.config import SeedConfig
from app.models import PlanStop, Vehicle
from common.config import ServiceSettings, load_dataset_config, load_section
from common.db import connect, insert_models

log = logging.getLogger(__name__)

# (schedule, traffic) per dataset part, relative to the dataset root.
PERIOD_FILES = {
    "train": ("train/schedule.csv", "train/traffic.csv"),
    "test": ("test/schedule.csv", "test/traffic.csv"),
    "validate": ("validate/schedule_plan.csv", "validate/traffic.csv"),
}

# The reference tables this loader owns. The demo reset keeps them, so a
# new one belongs here, not only in run_seed.
SEED_TABLES = ("stops_plan", "vehicles")

_POINT = re.compile(r"^\s*POINT\s*\(\s*(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s*\)\s*$")


def parse_point(value: str) -> tuple[float, float]:
    """``POINT (lon lat)`` as a ``(lon, lat)`` pair."""
    match = _POINT.match(value)
    if match is None:
        raise ValueError(f"not a WKT point: {value!r}")
    return float(match[1]), float(match[2])


def parse_id(value: str | None) -> int | None:
    """A dataset id as ``int``, or ``None`` if it is blank or not integral.

    Ids sometimes arrive as "123.0" after a trip through pandas with NaNs
    in the column. Parsed with ``Decimal``, not ``float``: ids can exceed
    2**53, past which a float silently loses precision.
    """
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    try:
        number = Decimal(text)
    except InvalidOperation:
        return None
    if number != number.to_integral_value():
        return None
    return int(number)


def parse_timestamp(value: str, tz: ZoneInfo) -> datetime:
    """Naive dataset timestamp, possibly with nanoseconds, as a UTC datetime.

    The dataset carries no zone; ``tz`` is the zone its timestamps are in.
    A value that already carries a UTC offset is rejected: it is not the
    naive format the dataset uses, so treating ``tz`` as its zone would be
    wrong.
    """
    text = value.strip()
    head, dot, fraction = text.partition(".")
    if dot:
        text = f"{head}.{fraction[:6]}"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is not None:
        raise ValueError(f"dataset timestamp already carries a UTC offset: {value!r}")
    return parsed.replace(tzinfo=tz).astimezone(UTC)


def _log_skipped(path: Path, reasons: dict[str, int]) -> None:
    if not reasons:
        return
    total = sum(reasons.values())
    detail = ", ".join(f"{count} bad {reason}" for reason, count in reasons.items())
    log.warning("skipped %d rows in %s: %s", total, path, detail)


def read_plan_stops(path: Path, tz: ZoneInfo) -> list[PlanStop]:
    """Planned arrivals from a schedule file; only plan columns are read.

    A row with a blank or unparsable id, a blank or unparsable ``geom``,
    or an unparsable ``time_begin`` is skipped rather than failing the
    whole file.
    """
    stops = []
    skipped: dict[str, int] = {}
    with path.open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            stop_id = parse_id(row.get("tt_action_item_id"))
            tr_id = parse_id(row.get("tr_id"))
            if stop_id is None or tr_id is None:
                skipped["id"] = skipped.get("id", 0) + 1
                continue
            geom = (row.get("geom") or "").strip()
            if not geom:
                skipped["geom"] = skipped.get("geom", 0) + 1
                continue
            try:
                lon, lat = parse_point(geom)
            except ValueError:
                skipped["geom"] = skipped.get("geom", 0) + 1
                continue
            try:
                time_plan = parse_timestamp(row["time_begin"], tz)
            except ValueError:
                skipped["time_begin"] = skipped.get("time_begin", 0) + 1
                continue
            stops.append(
                PlanStop(
                    stop_id=stop_id,
                    tr_id=tr_id,
                    time_plan=time_plan,
                    lat=lat,
                    lon=lon,
                    address=row.get("building_address") or None,
                )
            )
    _log_skipped(path, skipped)
    return stops


def read_vehicles(path: Path) -> list[Vehicle]:
    """Distinct terminal → vehicle pairs from a telemetry file.

    A row with a blank or unparsable id is skipped. When a unit maps to
    more than one vehicle across the file, the first mapping seen wins;
    conflicts are logged once per unit, after the file is read.
    """
    pairs: dict[int, int] = {}
    conflicts: dict[int, int] = {}
    skipped = 0
    with path.open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            unit_id = parse_id(row.get("unit_id"))
            tr_id = parse_id(row.get("tr_id"))
            if unit_id is None or tr_id is None:
                skipped += 1
                continue
            known = pairs.setdefault(unit_id, tr_id)
            if known != tr_id:
                conflicts[unit_id] = conflicts.get(unit_id, 0) + 1
    if skipped:
        _log_skipped(path, {"id": skipped})
    for unit_id, count in conflicts.items():
        log.warning(
            "unit %s maps to multiple vehicles; keeping %s (%d conflicting row(s))",
            unit_id,
            pairs[unit_id],
            count,
        )
    return [Vehicle(unit_id=u, tr_id=t) for u, t in pairs.items()]


def run_seed(
    conn: psycopg.Connection, data_dir: Path, config: SeedConfig, tz: ZoneInfo
) -> tuple[int, int]:
    """Replace the period's reference data; returns (stops, vehicles).

    A missing dataset touches nothing: the tables are left as they are.
    When the dataset is present, ``stops_plan`` and ``vehicles`` are
    cleared and reloaded from it in this same transaction, so a
    re-run never accumulates data across periods. Does not commit.
    """
    schedule_rel, traffic_rel = PERIOD_FILES[config.period]
    schedule, traffic = data_dir / schedule_rel, data_dir / traffic_rel
    missing = [str(p) for p in (schedule, traffic) if not p.is_file()]
    if missing:
        log.warning("dataset files not found, nothing loaded: %s", ", ".join(missing))
        return 0, 0

    stops = read_plan_stops(schedule, tz)
    if schedule.stat().st_size > 0 and not stops:
        raise ValueError(f"{schedule}: no valid rows in a non-empty file")
    vehicles = read_vehicles(traffic)
    if traffic.stat().st_size > 0 and not vehicles:
        raise ValueError(f"{traffic}: no valid rows in a non-empty file")

    conn.execute("DELETE FROM stops_plan")
    conn.execute("DELETE FROM vehicles")
    insert_models(conn, "stops_plan", stops)
    insert_models(conn, "vehicles", vehicles)
    return len(stops), len(vehicles)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    settings = ServiceSettings()
    config = load_section(settings.config_path, "seed", SeedConfig)
    dataset = load_dataset_config(settings.config_path)
    with connect(settings.database_url) as conn:
        stops, vehicles = run_seed(conn, settings.data_dir, config, dataset.zone)
        conn.commit()
    log.info("seed %s: %d planned stops, %d terminals", config.period, stops, vehicles)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
