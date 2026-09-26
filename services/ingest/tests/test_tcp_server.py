import asyncio
import logging
import struct

import pytest

from app.config import NdtpConfig
from app.ndtp import crc16_modbus, swapped_crc
from app.tcp_server import serve_client
from contracts import TelemetryRecord

UNIT = 664030


def frame(service: int, message: int, body: bytes) -> bytes:
    payload = struct.pack("<HHHI", service, message, 1, 1) + body
    header = struct.pack(
        "<HHHHBIH", 0x7E7E, len(payload), 0, swapped_crc(crc16_modbus(payload)), 2, UNIT, 0
    )
    return header + payload


HANDSHAKE = frame(0, 100, struct.pack("<HHHIII", 6, 2, 0, UNIT, 65535, 0))
NAV = struct.pack(
    "<IIIBBHHHHHBB", 1_788_000_000, 376173210, 557551234, 0xE0, 12, 28, 30, 180, 100, 50, 8, 2
)
REALTIME = frame(1, 101, bytes([0, 0]) + NAV)


class Writer:
    closed = False

    def close(self) -> None:
        self.closed = True

    async def wait_closed(self) -> None:
        pass


@pytest.mark.anyio
async def test_connection_accepts_handshake_then_realtime_and_closes_cleanly() -> None:
    reader = asyncio.StreamReader()
    reader.feed_data(HANDSHAKE[:7])
    reader.feed_data(HANDSHAKE[7:] + REALTIME)
    reader.feed_eof()
    writer = Writer()
    saved: list[tuple[str, TelemetryRecord]] = []

    async def shift_for(_timestamp: int) -> float:
        return 0

    async def save(source_key: str, record: TelemetryRecord) -> None:
        saved.append((source_key, record))

    await serve_client(reader, writer, NdtpConfig(), {UNIT: 115106}, shift_for=shift_for, save=save)
    assert len(saved) == 1
    assert saved[0][1].tr_id == 115106
    assert saved[0][1].lat == 55.7551234
    assert writer.closed


TAIL = frame(
    1, 101, bytes([0, 0]) + NAV + bytes([8, 0]) + bytes(6) + bytes([99, 0]) + b"\x01\x02\x03"
)


@pytest.mark.anyio
async def test_unknown_trailing_cell_keeps_the_point_and_is_logged_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    reader = asyncio.StreamReader()
    reader.feed_data(HANDSHAKE + TAIL + TAIL)
    reader.feed_eof()
    saved: list[tuple[str, TelemetryRecord]] = []

    async def shift_for(_timestamp: int) -> float:
        return 0

    async def save(source_key: str, record: TelemetryRecord) -> None:
        saved.append((source_key, record))

    with caplog.at_level(logging.INFO, logger="app.tcp_server"):
        await serve_client(
            reader, Writer(), NdtpConfig(), {UNIT: 115106}, shift_for=shift_for, save=save
        )
    assert len(saved) == 2
    assert saved[0][1].lat == 55.7551234
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "unknown_cell" in warnings[0] and "type 99" in warnings[0]
    infos = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
    assert any("unknown_cell type 99: 2" in message for message in infos)
