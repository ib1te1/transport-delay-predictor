from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from app.config import SeedConfig
from app.seed import parse_point, parse_timestamp, read_plan_stops, read_vehicles, run_seed

UTC_ZONE = ZoneInfo("UTC")

SCHEDULE_WITH_FACT = (
    "tt_action_item_id,time_begin,time_fact_begin,order_date,manual_fill,"
    "tr_id,geom,building_address\n"
    "10,2026-01-06 06:36:00.000000000,2026-01-06 06:39:01.000000000,2026-01-06,True,7,"
    'POINT (37.43070705 55.8040083),"Строгинское ш., д.1"\n'
    "11,2026-01-06 06:41:00.000000000,,2026-01-06,True,7,POINT (37.46 55.80),\n"
)
SCHEDULE_PLAN = (
    "tt_action_item_id,time_begin,order_date,manual_fill,tr_id,geom,building_address\n"
    '10,2026-01-06 06:36:00,2026-01-06,True,7,POINT (37.43 55.80),"адрес"\n'
)
TRAFFIC = (
    "packet_id,tr_id,unit_id,event_time,device_event_id,location_valid,gps_time,"
    "lon,lat,alt,speed,heading,receive_time,is_hist_data\n"
    "1,7,700,2026-01-06 12:30:31.462764,0,False,,,,,,,2026-01-06 12:30:31.462771,False\n"
    "2,7,700,2026-01-06 12:30:45.000000,0,True,,37.4,55.8,150,20,90,2026-01-06 12:30:45,False\n"
    "3,8,800,2026-01-06 12:30:45.000000,0,True,,37.4,55.8,150,20,90,2026-01-06 12:30:45,False\n"
)


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_parse_point_returns_lon_lat():
    assert parse_point("POINT (37.43070705 55.8040083)") == (37.43070705, 55.8040083)


def test_parse_point_rejects_garbage():
    with pytest.raises(ValueError):
        parse_point("LINESTRING (1 2, 3 4)")


def test_parse_timestamp_truncates_nanoseconds_and_sets_zone():
    parsed = parse_timestamp("2026-01-06 06:39:01.123456789", UTC_ZONE)
    assert parsed == datetime(2026, 1, 6, 6, 39, 1, 123456, tzinfo=UTC)


def test_read_plan_stops_never_carries_the_fact(tmp_path):
    stops = read_plan_stops(write(tmp_path / "s.csv", SCHEDULE_WITH_FACT), UTC_ZONE)

    assert [s.stop_id for s in stops] == [10, 11]
    assert "time_fact_begin" not in stops[0].model_dump()
    assert stops[0].time_plan == datetime(2026, 1, 6, 6, 36, tzinfo=UTC)
    assert (stops[0].lon, stops[0].lat) == (37.43070705, 55.8040083)
    assert stops[0].address == "Строгинское ш., д.1"
    assert stops[1].address is None


def test_read_plan_stops_accepts_plan_only_file(tmp_path):
    stops = read_plan_stops(write(tmp_path / "p.csv", SCHEDULE_PLAN), UTC_ZONE)
    assert [s.stop_id for s in stops] == [10]


def test_read_vehicles_deduplicates_pairs(tmp_path):
    vehicles = read_vehicles(write(tmp_path / "t.csv", TRAFFIC))
    assert sorted((v.unit_id, v.tr_id) for v in vehicles) == [(700, 7), (800, 8)]


def test_run_seed_loads_and_is_idempotent(db_conn, tmp_path):
    write(tmp_path / "test" / "schedule.csv", SCHEDULE_WITH_FACT)
    write(tmp_path / "test" / "traffic.csv", TRAFFIC)
    config = SeedConfig(period="test")

    assert run_seed(db_conn, tmp_path, config) == (2, 2)
    assert run_seed(db_conn, tmp_path, config) == (2, 2)

    stops = db_conn.execute("SELECT count(*) FROM stops_plan WHERE stop_id IN (10, 11)")
    assert stops.fetchone()[0] == 2
    vehicle = db_conn.execute("SELECT tr_id FROM vehicles WHERE unit_id = 800")
    assert vehicle.fetchone()[0] == 8


def test_run_seed_without_dataset_loads_nothing(db_conn, tmp_path):
    assert run_seed(db_conn, tmp_path / "absent", SeedConfig()) == (0, 0)
