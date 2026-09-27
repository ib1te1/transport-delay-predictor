"""Dashboard WebSocket: numbered updates, and the snapshot/stream handover of §7.4."""

import json
import time

import synthetic
from conftest import WS_URL, Replayed, Stack, parse_time
from websockets.sync.client import connect

TYPES = {"clock", "vehicles", "prediction", "alert"}


def _assert_consecutive(seqs: list[int]) -> None:
    assert seqs, "no messages"
    assert seqs == list(range(seqs[0], seqs[0] + len(seqs))), seqs


def test_updates_during_replay_are_numbered(replayed: Replayed) -> None:
    ws = replayed.ws
    assert ws.error is None, repr(ws.error)
    messages = ws.messages
    assert {m["type"] for m in messages} <= TYPES
    _assert_consecutive([m["data"]["seq"] for m in messages])
    kinds = {m["type"] for m in messages}
    assert {"clock", "vehicles", "prediction"} <= kinds, kinds


def test_prediction_and_vehicle_messages_during_replay(replayed: Replayed) -> None:
    buses = {b.tr_id for b in synthetic.BUSES}
    predictions = [m["data"] for m in replayed.ws.messages if m["type"] == "prediction"]
    assert {p["tr_id"] for p in predictions} == buses
    for data in predictions:
        assert data["prediction"]["sample_id"].startswith(f"{data['tr_id']}_")
    diffs = [m["data"] for m in replayed.ws.messages if m["type"] == "vehicles"]
    seen = {v["tr_id"] for d in diffs for v in d["vehicles"]}
    assert seen == buses | {synthetic.IDLE_TR_ID}
    clocks = [parse_time(m["data"]["clock"]) for m in replayed.ws.messages if m["type"] == "clock"]
    assert clocks == sorted(clocks)
    assert clocks[-1] == synthetic.last_event_time()


def test_snapshot_then_stream_without_gaps(stack: Stack, replayed: Replayed) -> None:
    """Open /ws, take the snapshot, drop what it covers, the rest follows it by one."""
    with connect(WS_URL, open_timeout=10) as ws:
        snapshot = stack.state()
        received = []
        deadline = time.monotonic() + 15
        while len(received) < 3 and time.monotonic() < deadline:
            message = json.loads(ws.recv(timeout=deadline - time.monotonic()))
            received.append(message)
    newer = [m["data"]["seq"] for m in received if m["data"]["seq"] > snapshot["seq"]]
    assert len(newer) >= 2, received
    assert newer[0] == snapshot["seq"] + 1
    _assert_consecutive(newer)
    # the clock keeps ticking once a second after the replay stopped
    assert {m["type"] for m in received} >= {"clock"}
