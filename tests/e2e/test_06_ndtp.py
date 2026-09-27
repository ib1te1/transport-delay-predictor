"""Scenario B: NDTP over TCP into ingest, the path the organizers' emulator takes.

Frames are built with the CRC and layouts of services/ingest/app/ndtp.py,
loaded by path: every service ships a package named ``app``, so it is not
imported as one. The idle terminal of the synthetic dataset reports three
new points over two connections, with a damaged frame in between.
"""

import importlib.util
import socket
import struct
from datetime import timedelta

import pytest
import synthetic
from conftest import INGEST_URL, NDTP_HOST, NDTP_PORT, ROOT, Replayed, Stack, parse_time, wait_for

_spec = importlib.util.spec_from_file_location(
    "ingest_ndtp", ROOT / "services" / "ingest" / "app" / "ndtp.py"
)
ndtp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ndtp)

UNIT = synthetic.IDLE_UNIT_ID
# Terminal clock of the first point; ingest maps it to NDTP_ANCHOR.
TS0 = 1_800_000_000
# (seconds after TS0, lat, lon, speed, course)
POINTS = [
    (0, 55.7010000, 37.5010000, 21, 45),
    (5, 55.7020000, 37.5020000, 22, 46),
    (10, 55.7030000, 37.5030000, 23, 47),
]
# latitude and longitude positive, fix valid (bits 5, 6, 7)
VALID_FIX = (1 << 5) | (1 << 6) | (1 << 7)


def frame(service: int, message: int, body: bytes, *, unit: int = UNIT) -> bytes:
    payload = struct.pack(ndtp.NPH_FORMAT, service, message, 1, 1) + body
    crc = ndtp.swapped_crc(ndtp.crc16_modbus(payload))
    header = struct.pack(
        ndtp.NPL_FORMAT, ndtp.SIGNATURE, len(payload), 0, crc, ndtp.NPH_PACKET, unit, 0
    )
    return header + payload


def handshake(unit: int = UNIT) -> bytes:
    body = struct.pack("<HHHIII", 6, 2, 0, unit, 65535, 0)
    return frame(ndtp.HANDSHAKE_SERVICE, ndtp.HANDSHAKE_TYPE, body, unit=unit)


def navigation(offset: int, lat: float, lon: float, speed: int, course: int) -> bytes:
    nav = struct.pack(
        ndtp.NAV_FORMAT,
        TS0 + offset,
        round(lon * 1e7),
        round(lat * 1e7),
        VALID_FIX,
        12,
        speed,
        30,
        course,
        100,
        50,
        8,
        2,
    )
    # a known auxiliary cell after the navigation, as the emulator sends
    body = bytes([0, 0]) + nav + bytes([8, 0]) + bytes(6)
    return frame(ndtp.REALTIME_SERVICE, ndtp.REALTIME_TYPE, body)


def send_session(*frames: bytes) -> None:
    """One terminal connection: send, then close and let ingest see EOF."""
    with socket.create_connection((NDTP_HOST, NDTP_PORT), timeout=10) as sock:
        for data in frames:
            sock.sendall(data)
        sock.shutdown(socket.SHUT_WR)
        # ingest closes its side after EOF; waiting for it keeps the order of sessions
        sock.settimeout(15)
        while sock.recv(1024):
            pass


def _emulator_rows(stack: Stack) -> list[list[str]]:
    return stack.compose.sql(
        "SELECT tr_id, event_time AT TIME ZONE 'UTC', lat, lon, speed_kmh, heading_deg"
        f" FROM telemetry WHERE unit_id = {UNIT} AND source = 'emulator' ORDER BY id"
    )


def test_garbage_is_rejected_without_taking_the_listener_down(
    stack: Stack, replayed: Replayed
) -> None:
    before = stack.compose.count("telemetry")
    # returns only once ingest has closed the connection on the bad header
    send_session(b"not an ndtp frame at all")
    assert stack.json(f"{INGEST_URL}/health")["status"] == "ok"
    assert stack.compose.count("telemetry") == before


def test_ndtp_frames_land_in_telemetry(stack: Stack, replayed: Replayed) -> None:
    before = len(_emulator_rows(stack))
    damaged = bytearray(navigation(*POINTS[1]))
    damaged[-1] ^= 0xFF
    first, second, third = (navigation(*p) for p in POINTS)
    send_session(handshake(), first, bytes(damaged), second)
    # a reconnect with a new handshake is the normal path
    send_session(handshake(), third)

    rows = wait_for(
        lambda: (r := _emulator_rows(stack)) and len(r) >= before + len(POINTS) and r,
        "NDTP points in the telemetry table",
        30,
    )
    new = rows[before:]
    assert len(new) == len(POINTS), new
    for row, (offset, lat, lon, speed, course) in zip(new, POINTS, strict=True):
        tr_id, event_time, *measured = row
        assert int(tr_id) == synthetic.IDLE_TR_ID
        expected_time = synthetic.NDTP_ANCHOR + timedelta(seconds=offset)
        assert parse_time(event_time + "+00:00") == expected_time
        assert [float(v) for v in measured] == pytest.approx([lat, lon, speed, course])
    assert stack.compose.count("telemetry", "published_at IS NULL") == 0


def test_ndtp_points_reach_the_stream_and_the_map(stack: Stack, replayed: Replayed) -> None:
    records = [r for r in stack.compose.stream("telemetry") if r["source"] == "emulator"]
    assert records and {r["unit_id"] for r in records} == {UNIT}
    last_offset, lat, lon, _speed, _course = POINTS[-1]
    last_seen = synthetic.NDTP_ANCHOR + timedelta(seconds=last_offset)

    def moved() -> dict | None:
        state = stack.state()
        view = next(v for v in state["vehicles"] if v["tr_id"] == synthetic.IDLE_TR_ID)
        return state if parse_time(view["last_seen"]) == last_seen else None

    state = wait_for(moved, "the idle terminal to move on the map", 30)
    view = next(v for v in state["vehicles"] if v["tr_id"] == synthetic.IDLE_TR_ID)
    assert (view["lat"], view["lon"]) == pytest.approx((lat, lon))
    assert view["freshness"] == "active"
    assert parse_time(state["clock"]) == last_seen
