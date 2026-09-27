"""Scenario A: CSV replay lands in Postgres, on the telemetry stream and on the map."""

from datetime import timedelta

import synthetic
from conftest import API_URL, Replayed, Stack, parse_time


def test_replay_run_completed(stack: Stack, replayed: Replayed) -> None:
    rows = stack.compose.sql(
        "SELECT status, period, speedup FROM ingest_runs WHERE mode = 'replay'"
    )
    assert rows == [["completed", synthetic.PERIOD, "240"]]
    assert "replay complete" in replayed.replay_output


def test_telemetry_stored_and_published(stack: Stack, replayed: Replayed) -> None:
    compose = stack.compose
    fixes = synthetic.all_fixes()
    assert compose.count("telemetry", "source = 'replay'") == len(fixes)
    assert compose.count("telemetry", "published_at IS NULL OR stream_id IS NULL") == 0
    assert compose.xlen("telemetry") == len(fixes)


def test_telemetry_rows_match_the_dataset(stack: Stack, replayed: Replayed) -> None:
    for bus in synthetic.BUSES:
        expected = [(f.event_time, f.lat, f.lon, f.speed) for f in synthetic.fixes(bus)]
        rows = stack.compose.sql(
            "SELECT tr_id, event_time AT TIME ZONE 'UTC', lat, lon, speed_kmh FROM telemetry"
            f" WHERE unit_id = {bus.unit_id} ORDER BY event_time"
        )
        assert {int(r[0]) for r in rows} == {bus.tr_id}
        got = [(parse_time(r[1] + "+00:00"), float(r[2]), float(r[3]), float(r[4])) for r in rows]
        assert got == expected


def test_telemetry_stream_is_in_dataset_time_order(stack: Stack, replayed: Replayed) -> None:
    records = stack.compose.stream("telemetry")
    times = [parse_time(r["event_time"]) for r in records]
    assert times == sorted(times)
    assert times[0] == synthetic.first_event_time()
    assert times[-1] == synthetic.last_event_time()
    assert {r["source"] for r in records} == {"replay"}


def _expected_freshness(silence: timedelta) -> str:
    # api.stale_after_sec and api.drop_after_sec in the e2e config
    if silence <= timedelta(seconds=120):
        return "active"
    if silence <= timedelta(seconds=900):
        return "stale"
    return "offline"


def test_snapshot_shows_every_vehicle_at_its_last_position(
    stack: Stack, replayed: Replayed
) -> None:
    state = stack.state()
    clock = synthetic.last_event_time()
    assert parse_time(state["clock"]) == clock
    assert state["seq"] > 0
    vehicles = {v["tr_id"]: v for v in state["vehicles"]}
    assert list(vehicles) == sorted(vehicles)
    last_fixes = {b.tr_id: synthetic.fixes(b)[-1] for b in synthetic.BUSES}
    last_fixes[synthetic.IDLE_TR_ID] = synthetic.idle_fix()
    assert set(vehicles) == set(last_fixes)
    for tr_id, fix in last_fixes.items():
        view = vehicles[tr_id]
        assert view["unit_id"] == fix.unit_id
        assert (view["lat"], view["lon"]) == (fix.lat, fix.lon)
        assert view["speed_kmh"] == fix.speed
        assert parse_time(view["last_seen"]) == fix.event_time
        assert view["freshness"] == _expected_freshness(clock - fix.event_time), view
    freshness = state["summary"]["freshness"]
    assert sum(freshness.values()) == len(vehicles)
    for level, count in freshness.items():
        assert count == sum(v["freshness"] == level for v in vehicles.values())


def test_rerunning_a_completed_replay_is_refused(stack: Stack, replayed: Replayed) -> None:
    result = stack.compose.run_replay()
    output = result.stdout + result.stderr
    assert result.returncode == 3, output
    assert "scripts/reset-demo" in output
    assert stack.compose.count("telemetry") == len(synthetic.all_fixes())


def test_card_of_an_unknown_vehicle_is_404(stack: Stack, replayed: Replayed) -> None:
    assert stack.get(f"{API_URL}/api/vehicles/999999").status_code == 404
