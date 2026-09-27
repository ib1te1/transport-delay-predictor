"""Tiny deterministic dataset for the end-to-end suite.

Three buses drive east along their own straight street, one per vehicle,
with a planned stop every 120 s and ~330 m. Each bus reaches every stop
the same number of seconds off the plan, stands there 30 s at speed 0 and
drives on at ~13 km/h, a fix every 10 s. So the matcher must see one
arrival per visited stop with exactly that delay.

A fourth terminal sends one point at the start and has no plan: it is on
the map but never scored. The NDTP test moves it later.

Run ``python tests/e2e/synthetic.py`` to rewrite the CSV files under
``tests/e2e/data/validate``. The tests import the constants from here.
"""

import csv
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
PERIOD = "validate"

START = datetime(2026, 1, 6, 6, 0, tzinfo=UTC)
STOP_EVERY = timedelta(seconds=120)
DWELL_SEC = 30
FIX_EVERY_SEC = 10
# Stops each bus actually passes; the plan goes further so that a target
# stop 10-15 minutes ahead exists at every tick.
VISITED_STOPS = 20
PLANNED_STOPS = 35
BASE_LAT = 55.75
BASE_LON = 37.60
# ~330 m between stops at this latitude, far more than the 50 m radius.
STOP_DLON = 0.005
STREET_DLAT = 0.01


@dataclass(frozen=True)
class Bus:
    tr_id: int
    unit_id: int
    delay_s: int
    street: int


BUSES = (
    Bus(tr_id=101, unit_id=9101, delay_s=60, street=0),
    Bus(tr_id=102, unit_id=9102, delay_s=150, street=1),
    Bus(tr_id=103, unit_id=9103, delay_s=-45, street=2),
)

# Seeded from traffic.csv like the rest, but without planned stops.
IDLE_TR_ID = 104
IDLE_UNIT_ID = 9104
IDLE_LAT = 55.70
IDLE_LON = 37.50
# ingest.ndtp.dataset_anchor in tests/e2e/config/ingest-ndtp.yaml: the
# first NDTP point maps here, a minute after the last CSV point.
NDTP_ANCHOR = datetime(2026, 1, 6, 6, 42, tzinfo=UTC)


def stop_id(bus: Bus, k: int) -> int:
    return bus.tr_id * 1000 + k


def stop_lat(bus: Bus) -> float:
    return round(BASE_LAT + bus.street * STREET_DLAT, 7)


def stop_lon(k: int) -> float:
    return round(BASE_LON + k * STOP_DLON, 7)


def time_plan(k: int) -> datetime:
    return START + k * STOP_EVERY


def arrival(bus: Bus, k: int) -> datetime:
    return time_plan(k) + timedelta(seconds=bus.delay_s)


def address(bus: Bus, k: int) -> str:
    return f"E2E street {bus.street}, stop {k}"


@dataclass(frozen=True)
class Fix:
    unit_id: int
    tr_id: int
    event_time: datetime
    lat: float
    lon: float
    speed: float
    heading: float


def fixes(bus: Bus) -> list[Fix]:
    """Stand at stop k for DWELL_SEC, then drive to stop k + 1, one fix every 10 s."""
    lat = stop_lat(bus)
    points = []
    step = timedelta(seconds=FIX_EVERY_SEC)
    drive = STOP_EVERY - timedelta(seconds=DWELL_SEC)
    for k in range(VISITED_STOPS):
        t = arrival(bus, k)
        leave = t + timedelta(seconds=DWELL_SEC)
        last = k == VISITED_STOPS - 1
        end = leave if last else arrival(bus, k + 1)
        while t < end or (last and t <= end):
            if t <= leave:
                lon, speed = stop_lon(k), 0.0
            else:
                share = (t - leave) / drive
                lon = stop_lon(k) + (stop_lon(k + 1) - stop_lon(k)) * share
                speed = 13.0
            points.append(Fix(bus.unit_id, bus.tr_id, t, lat, round(lon, 7), speed, 90.0))
            t += step
    return points


def idle_fix() -> Fix:
    return Fix(IDLE_UNIT_ID, IDLE_TR_ID, START, IDLE_LAT, IDLE_LON, 0.0, 0.0)


def all_fixes() -> list[Fix]:
    return [idle_fix(), *(f for bus in BUSES for f in fixes(bus))]


def last_event_time() -> datetime:
    return max(f.event_time for f in all_fixes())


def first_event_time() -> datetime:
    return min(f.event_time for f in all_fixes())


def _naive(value: datetime) -> str:
    return value.astimezone(UTC).strftime("%Y-%m-%d %H:%M:%S")


def write(root: Path = DATA_DIR) -> None:
    folder = root / PERIOD
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "schedule_plan.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(["tt_action_item_id", "tr_id", "geom", "time_begin", "building_address"])
        for bus in BUSES:
            for k in range(PLANNED_STOPS):
                writer.writerow(
                    [
                        stop_id(bus, k),
                        bus.tr_id,
                        f"POINT ({stop_lon(k)} {stop_lat(bus)})",
                        _naive(time_plan(k)),
                        address(bus, k),
                    ]
                )
    with (folder / "traffic.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(
            ["unit_id", "tr_id", "event_time", "location_valid", "lat", "lon", "speed", "heading"]
        )
        # Grouped by vehicle, not by time: replay has to sort it.
        for fix in all_fixes():
            writer.writerow(
                [
                    fix.unit_id,
                    fix.tr_id,
                    _naive(fix.event_time),
                    "True",
                    fix.lat,
                    fix.lon,
                    fix.speed,
                    fix.heading,
                ]
            )


if __name__ == "__main__":
    write()
    print(f"wrote {len(all_fixes())} fixes to {DATA_DIR / PERIOD}")
