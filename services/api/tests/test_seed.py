import logging
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from app.config import SeedConfig
from app.seed import (
    parse_id,
    parse_point,
    parse_timestamp,
    read_plan_stops,
    read_vehicles,
    run_seed,
)

UTC_ZONE = ZoneInfo("UTC")
MOSCOW = ZoneInfo("Europe/Moscow")

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
SCHEDULE_WITH_BAD_ROWS = (
    "tt_action_item_id,time_begin,order_date,manual_fill,tr_id,geom,building_address\n"
    '10,2026-01-06 06:36:00,2026-01-06,True,7,POINT (37.43 55.80),"адрес"\n'
    "11,2026-01-06 06:41:00,2026-01-06,True,7,,\n"
    "12,not-a-time,2026-01-06,True,7,POINT (37.46 55.80),\n"
)
SCHEDULE_ONLY_BAD_ROWS = (
    "tt_action_item_id,time_begin,order_date,manual_fill,tr_id,geom,building_address\n"
    "11,2026-01-06 06:41:00,2026-01-06,True,7,,\n"
    "12,not-a-time,2026-01-06,True,7,POINT (37.46 55.80),\n"
)
TRAFFIC = (
    "packet_id,tr_id,unit_id,event_time,device_event_id,location_valid,gps_time,"
    "lon,lat,alt,speed,heading,receive_time,is_hist_data\n"
    "1,7,700,2026-01-06 12:30:31.462764,0,False,,,,,,,2026-01-06 12:30:31.462771,False\n"
    "2,7,700,2026-01-06 12:30:45.000000,0,True,,37.4,55.8,150,20,90,2026-01-06 12:30:45,False\n"
    "3,8,800,2026-01-06 12:30:45.000000,0,True,,37.4,55.8,150,20,90,2026-01-06 12:30:45,False\n"
)
TRAFFIC_WITH_DECIMAL_IDS = (
    "packet_id,tr_id,unit_id,event_time,device_event_id,location_valid,gps_time,"
    "lon,lat,alt,speed,heading,receive_time,is_hist_data\n"
    "1,7,700.0,2026-01-06 12:30:31.462764,0,False,,,,,,,2026-01-06 12:30:31.462771,False\n"
    "2,7,700.5,2026-01-06 12:30:45.000000,0,True,,37.4,55.8,150,20,90,2026-01-06 12:30:45,False\n"
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


def test_parse_id_accepts_a_plain_integer():
    assert parse_id("700") == 700


def test_parse_id_accepts_an_integral_decimal():
    assert parse_id("700.0") == 700


def test_parse_id_rejects_a_non_integral_decimal():
    assert parse_id("700.5") is None


def test_parse_id_rejects_blank_and_garbage():
    assert parse_id("") is None
    assert parse_id(None) is None
    assert parse_id("abc") is None


def test_parse_timestamp_truncates_nanoseconds_and_sets_zone():
    parsed = parse_timestamp("2026-01-06 06:39:01.123456789", UTC_ZONE)
    assert parsed == datetime(2026, 1, 6, 6, 39, 1, 123456, tzinfo=UTC)


def test_parse_timestamp_converts_the_source_zone_to_utc():
    parsed = parse_timestamp("2026-01-06 06:39:01", MOSCOW)
    assert parsed == datetime(2026, 1, 6, 3, 39, 1, tzinfo=UTC)
    assert parsed.tzinfo is UTC


def test_parse_timestamp_rejects_a_value_that_already_carries_an_offset():
    with pytest.raises(ValueError, match="offset"):
        parse_timestamp("2026-01-06T06:39:01+03:00", UTC_ZONE)


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


def test_read_plan_stops_skips_bad_geom_and_bad_time(tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        stops = read_plan_stops(write(tmp_path / "s.csv", SCHEDULE_WITH_BAD_ROWS), UTC_ZONE)

    assert [s.stop_id for s in stops] == [10]
    [record] = [r for r in caplog.records if "skipped" in r.message]
    assert "skipped 2 rows" in record.message
    assert "1 bad geom" in record.message
    assert "1 bad time_begin" in record.message


def test_read_plan_stops_accepts_decimal_ids(tmp_path):
    schedule = (
        "tt_action_item_id,time_begin,order_date,manual_fill,tr_id,geom,building_address\n"
        '700.0,2026-01-06 06:36:00,2026-01-06,True,7,POINT (37.43 55.80),"адрес"\n'
        "700.5,2026-01-06 06:36:00,2026-01-06,True,7,POINT (37.43 55.80),\n"
    )
    stops = read_plan_stops(write(tmp_path / "s.csv", schedule), UTC_ZONE)
    assert [s.stop_id for s in stops] == [700]


def test_read_vehicles_deduplicates_pairs(tmp_path):
    vehicles = read_vehicles(write(tmp_path / "t.csv", TRAFFIC))
    assert sorted((v.unit_id, v.tr_id) for v in vehicles) == [(700, 7), (800, 8)]


def test_read_vehicles_accepts_decimal_ids_and_skips_non_integral(tmp_path):
    vehicles = read_vehicles(write(tmp_path / "t.csv", TRAFFIC_WITH_DECIMAL_IDS))
    assert [(v.unit_id, v.tr_id) for v in vehicles] == [(700, 7)]


def test_read_vehicles_logs_conflicts_once_per_unit(tmp_path, caplog):
    traffic = (
        "packet_id,tr_id,unit_id,event_time,device_event_id,location_valid,gps_time,"
        "lon,lat,alt,speed,heading,receive_time,is_hist_data\n"
        "1,7,700,t,0,False,,,,,,,t,False\n"
        "2,8,700,t,0,False,,,,,,,t,False\n"
        "3,9,700,t,0,False,,,,,,,t,False\n"
    )
    with caplog.at_level(logging.WARNING):
        vehicles = read_vehicles(write(tmp_path / "t.csv", traffic))

    assert [(v.unit_id, v.tr_id) for v in vehicles] == [(700, 7)]
    conflict_records = [r for r in caplog.records if "conflicting" in r.message]
    assert len(conflict_records) == 1
    assert "2 conflicting" in conflict_records[0].message


def test_run_seed_loads(db_conn, tmp_path):
    write(tmp_path / "test" / "schedule.csv", SCHEDULE_WITH_FACT)
    write(tmp_path / "test" / "traffic.csv", TRAFFIC)
    config = SeedConfig(period="test")

    assert run_seed(db_conn, tmp_path, config, UTC_ZONE) == (2, 2)

    stops = db_conn.execute("SELECT count(*) FROM stops_plan WHERE stop_id IN (10, 11)")
    assert stops.fetchone()[0] == 2
    vehicle = db_conn.execute("SELECT tr_id FROM vehicles WHERE unit_id = 800")
    assert vehicle.fetchone()[0] == 8


def test_run_seed_without_dataset_loads_nothing(db_conn, tmp_path):
    assert run_seed(db_conn, tmp_path / "absent", SeedConfig(), UTC_ZONE) == (0, 0)


def test_run_seed_replaces_reference_data_instead_of_accumulating(db_conn, tmp_path):
    write(tmp_path / "test" / "schedule.csv", SCHEDULE_PLAN)
    write(tmp_path / "test" / "traffic.csv", TRAFFIC)
    run_seed(db_conn, tmp_path, SeedConfig(period="test"), UTC_ZONE)

    validate_schedule = (
        "tt_action_item_id,time_begin,order_date,manual_fill,tr_id,geom,building_address\n"
        '20,2026-01-06 06:36:00,2026-01-06,True,9,POINT (37.50 55.90),"адрес"\n'
    )
    validate_traffic = (
        "packet_id,tr_id,unit_id,event_time,device_event_id,location_valid,gps_time,"
        "lon,lat,alt,speed,heading,receive_time,is_hist_data\n"
        "1,9,900,2026-01-06 12:30:45.000000,0,True,,37.4,55.8,150,20,90,2026-01-06 12:30:45,False\n"
    )
    write(tmp_path / "validate" / "schedule_plan.csv", validate_schedule)
    write(tmp_path / "validate" / "traffic.csv", validate_traffic)

    assert run_seed(db_conn, tmp_path, SeedConfig(period="validate"), UTC_ZONE) == (1, 1)

    stop_ids = db_conn.execute("SELECT stop_id FROM stops_plan").fetchall()
    assert stop_ids == [(20,)]
    vehicle_pairs = db_conn.execute("SELECT unit_id, tr_id FROM vehicles").fetchall()
    assert vehicle_pairs == [(900, 9)]


def test_run_seed_raises_when_a_non_empty_file_yields_no_valid_rows(db_conn, tmp_path):
    write(tmp_path / "test" / "schedule.csv", SCHEDULE_ONLY_BAD_ROWS)
    write(tmp_path / "test" / "traffic.csv", TRAFFIC)

    with pytest.raises(ValueError, match="no valid rows"):
        run_seed(db_conn, tmp_path, SeedConfig(period="test"), UTC_ZONE)
