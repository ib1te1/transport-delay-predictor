import asyncio
import struct

import pytest

from app.ndtp import (
    NdtpError,
    crc16_modbus,
    parse_handshake,
    parse_realtime,
    read_frame,
    swapped_crc,
)

UNIT = 664030
NAV = struct.pack(
    "<IIIBBHHHHHBB",
    1_788_000_000,
    376173210,
    557551234,
    (1 << 5) | (1 << 6) | (1 << 7),
    12,
    28,
    30,
    180,
    100,
    50,
    8,
    2,
)


def frame(service: int, message: int, body: bytes, *, unit: int = UNIT) -> bytes:
    payload = struct.pack("<HHHI", service, message, 1, 1) + body
    header = struct.pack(
        "<HHHHBIH", 0x7E7E, len(payload), 0, swapped_crc(crc16_modbus(payload)), 2, unit, 0
    )
    return header + payload


HANDSHAKE = frame(0, 100, struct.pack("<HHHIII", 6, 2, 0, UNIT, 65535, 0))
REALTIME = frame(1, 101, bytes([0, 0]) + NAV + bytes([8, 0]) + bytes(6))


def test_modbus_crc_known_vector() -> None:
    assert crc16_modbus(b"123456789") == 0x4B37


@pytest.mark.anyio
async def test_fragmented_handshake_and_realtime_share_one_tcp_stream() -> None:
    reader = asyncio.StreamReader()
    for part in (HANDSHAKE[:3], HANDSHAKE[3:15], HANDSHAKE[15:] + REALTIME):
        reader.feed_data(part)
    first = await read_frame(reader, max_frame_bytes=65535, timeout_sec=1)
    assert parse_handshake(first) == UNIT
    second = await read_frame(reader, max_frame_bytes=65535, timeout_sec=1)
    nav = parse_realtime(second, expected_unit_id=UNIT)
    assert nav.longitude == 376173210
    assert nav.latitude == 557551234
    assert nav.speed_avg == 28
    assert nav.course == 180


@pytest.mark.anyio
async def test_bad_crc_drops_only_that_frame() -> None:
    reader = asyncio.StreamReader()
    bad = bytearray(REALTIME)
    bad[-1] ^= 1
    reader.feed_data(bytes(bad) + REALTIME)
    with pytest.raises(NdtpError, match="crc") as caught:
        await read_frame(reader, max_frame_bytes=65535, timeout_sec=1)
    assert not caught.value.fatal
    good = await read_frame(reader, max_frame_bytes=65535, timeout_sec=1)
    assert parse_realtime(good, expected_unit_id=UNIT).timestamp == 1_788_000_000


@pytest.mark.anyio
async def test_invalid_header_is_fatal() -> None:
    reader = asyncio.StreamReader()
    bad = bytearray(REALTIME)
    bad[:2] = b"xx"
    reader.feed_data(bad)
    with pytest.raises(NdtpError, match="signature") as caught:
        await read_frame(reader, max_frame_bytes=65535, timeout_sec=1)
    assert caught.value.fatal


@pytest.mark.anyio
async def test_wrong_handshake_address_is_rejected() -> None:
    bad = frame(0, 100, struct.pack("<HHHIII", 6, 2, 0, UNIT + 1, 65535, 0))
    reader = asyncio.StreamReader()
    reader.feed_data(bad)
    parsed = await read_frame(reader, max_frame_bytes=65535, timeout_sec=1)
    with pytest.raises(NdtpError, match="handshake_address"):
        parse_handshake(parsed)


@pytest.mark.anyio
async def test_unknown_auxiliary_cell_is_rejected() -> None:
    reader = asyncio.StreamReader()
    reader.feed_data(frame(1, 101, bytes([0, 0]) + NAV + bytes([99, 0])))
    parsed = await read_frame(reader, max_frame_bytes=65535, timeout_sec=1)
    with pytest.raises(NdtpError, match="unknown_cell"):
        parse_realtime(parsed, expected_unit_id=UNIT)
