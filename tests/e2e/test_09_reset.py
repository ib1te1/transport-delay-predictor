"""Demo reset (integration-design.md §5) and a second run after it.

Runs last: it wipes what the other scenarios checked. The second replay
also covers the regression where matcher emitted nothing after a reset.
"""

import synthetic
from conftest import API_URL, MATCHER_URL, Replayed, Stack, expected_stop_events, wait_settled

RUN_TABLES = (
    "telemetry",
    "stop_events",
    "predictions",
    "alerts",
    "matcher_cursor",
    "matcher_state",
    "matcher_outbox",
)
STREAMS = ("telemetry", "stop_events", "predictions")


def test_reset_clears_the_run_and_keeps_reference_data(stack: Stack, replayed: Replayed) -> None:
    compose = stack.compose
    plan, vehicles = compose.count("stops_plan"), compose.count("vehicles")
    assert compose.count("stop_events") > 0

    result = compose.reset_demo()
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "Demo reset" in output

    for table in RUN_TABLES:
        assert compose.count(table) == 0, table
    # ingest in NDTP mode opens an emulator run again as soon as it is back
    assert compose.count("ingest_runs", "mode = 'replay'") == 0
    for stream in STREAMS:
        assert compose.xlen(stream) == 0, stream
    assert compose.count("stops_plan") == plan
    assert compose.count("vehicles") == vehicles

    stack.wait_ready()
    # api starts over: no clock and nothing on the map until telemetry comes
    state = stack.state()
    assert state["clock"] is None and state["vehicles"] == []
    quality = stack.json(f"{MATCHER_URL}/quality")
    assert quality["stream_cursor"] == "0-0" and quality["stop_events"] == 0


def test_replay_after_reset_runs_the_whole_pipeline_again(stack: Stack) -> None:
    result = stack.compose.run_replay()
    assert result.returncode == 0, result.stdout + result.stderr
    settled = wait_settled(stack)
    assert settled["quality"]["stop_events"] == expected_stop_events()
    assert stack.compose.count("stop_events") == expected_stop_events()
    views = {v["tr_id"]: v for v in stack.state()["vehicles"]}
    for bus in synthetic.BUSES:
        assert views[bus.tr_id]["prediction"] is not None
        card = stack.json(f"{API_URL}/api/vehicles/{bus.tr_id}")
        assert any(s["delay_s"] == bus.delay_s for s in card["stops"])
