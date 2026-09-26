"""Framing and decoding for the NDTP subset emitted by the supplied emulator."""

import asyncio
import struct
from dataclasses import dataclass

NPL_SIZE = 15
NPH_SIZE = 10
NAV_SIZE = 26
NPL_FORMAT = "<HHHHBIH"
NPH_FORMAT = "<HHHI"
NAV_FORMAT = "<IIIBBHHHHHBB"
SIGNATURE = 0x7E7E
NPH_PACKET = 0x02
HANDSHAKE_SERVICE = 0
HANDSHAKE_TYPE = 100
REALTIME_SERVICE = 1
REALTIME_TYPE = 101
CELL_SIZES = {0: NAV_SIZE, 2: 26, 8: 6, 10: 37, 15: 50, 16: 8}


class NdtpError(ValueError):
    """A bad NDTP frame; ``fatal`` means its next boundary is unknown."""

    def __init__(self, reason: str, *, fatal: bool = False) -> None:
        super().__init__(reason)
        self.reason = reason
        self.fatal = fatal


@dataclass(frozen=True)
class Npl:
    data_size: int
    peer_address: int
    crc: int


@dataclass(frozen=True)
class Frame:
    unit_id: int
    service_id: int
    message_type: int
    flags: int
    request_id: int
    body: bytes


@dataclass(frozen=True)
class Nav:
    timestamp: int
    longitude: int
    latitude: int
    extra_dop: int
    speed_avg: int
    course: int


@dataclass(frozen=True)
class CellStop:
    """Where the auxiliary cell walk stopped; the navigation before it is kept.

    ``offset`` is the start of the offending cell header within the realtime body.
    """

    reason: str
    cell_type: int
    offset: int


@dataclass(frozen=True)
class Realtime:
    nav: Nav
    stop: CellStop | None = None


def crc16_modbus(data: bytes) -> int:
    crc = 0xFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ (0xA001 if crc & 1 else 0)
    return crc


def swapped_crc(value: int) -> int:
    return ((value & 0xFF) << 8) | (value >> 8)


def parse_npl(data: bytes, *, max_frame_bytes: int) -> Npl:
    if len(data) != NPL_SIZE:
        raise NdtpError("short_npl", fatal=True)
    signature, data_size, flags, crc, packet_type, peer_address, _request_id = struct.unpack(
        NPL_FORMAT, data
    )
    if signature != SIGNATURE:
        raise NdtpError("signature", fatal=True)
    if packet_type != NPH_PACKET or flags != 0:
        raise NdtpError("npl_type_or_flags", fatal=True)
    if data_size < NPH_SIZE or NPL_SIZE + data_size > max_frame_bytes:
        raise NdtpError("frame_size", fatal=True)
    return Npl(data_size, peer_address, crc)


def parse_frame(header: Npl, payload: bytes) -> Frame:
    if len(payload) != header.data_size:
        raise NdtpError("short_payload")
    if header.crc != swapped_crc(crc16_modbus(payload)):
        raise NdtpError("crc")
    service_id, message_type, flags, request_id = struct.unpack_from(NPH_FORMAT, payload)
    if not flags & 1:
        raise NdtpError("nph_flags")
    return Frame(
        header.peer_address, service_id, message_type, flags, request_id, payload[NPH_SIZE:]
    )


async def read_frame(
    reader: asyncio.StreamReader, *, max_frame_bytes: int, timeout_sec: float
) -> Frame:
    """Read exactly one frame regardless of TCP read boundaries."""
    npl_bytes = await asyncio.wait_for(reader.readexactly(NPL_SIZE), timeout_sec)
    header = parse_npl(npl_bytes, max_frame_bytes=max_frame_bytes)
    payload = await asyncio.wait_for(reader.readexactly(header.data_size), timeout_sec)
    return parse_frame(header, payload)


def parse_handshake(frame: Frame) -> int:
    """Return the terminal id after validating the emulator handshake."""
    if frame.service_id != HANDSHAKE_SERVICE or frame.message_type != HANDSHAKE_TYPE:
        raise NdtpError("expected_handshake", fatal=True)
    if len(frame.body) != 18:
        raise NdtpError("handshake_size", fatal=True)
    major, minor, flags, body_address, max_packet, reserved = struct.unpack("<HHHIII", frame.body)
    if (major, minor) != (6, 2) or flags != 0 or reserved != 0 or max_packet == 0:
        raise NdtpError("handshake_fields", fatal=True)
    if body_address != frame.unit_id:
        raise NdtpError("handshake_address", fatal=True)
    return frame.unit_id


def parse_realtime(frame: Frame, *, expected_unit_id: int) -> Realtime:
    """Extract the leading G6CellNav00 and walk the known auxiliary cells after it.

    The CRC already covers the whole body, so a trailing cell the decoder
    cannot size means an unknown or differently described cell, not a
    damaged navigation. The walk stops there and the navigation is kept.
    """
    if frame.service_id != REALTIME_SERVICE or frame.message_type != REALTIME_TYPE:
        raise NdtpError("expected_realtime")
    if frame.unit_id != expected_unit_id:
        raise NdtpError("peer_address", fatal=True)
    body = memoryview(frame.body)
    if len(body) < 2 + NAV_SIZE or body[0] != 0 or body[1] != 0:
        raise NdtpError("missing_nav")
    fields = struct.unpack_from(NAV_FORMAT, body, 2)
    nav = Nav(
        timestamp=fields[0],
        longitude=fields[1],
        latitude=fields[2],
        extra_dop=fields[3],
        speed_avg=fields[5],
        course=fields[7],
    )
    position = 2 + NAV_SIZE
    while position < len(body):
        cell_type = body[position]
        if len(body) - position < 2:
            return Realtime(nav, CellStop("short_cell_header", cell_type, position))
        if cell_type == 0 or cell_type not in CELL_SIZES:
            return Realtime(nav, CellStop("unknown_cell", cell_type, position))
        end = position + 2 + CELL_SIZES[cell_type]
        if end > len(body):
            return Realtime(nav, CellStop("short_cell", cell_type, position))
        position = end
    return Realtime(nav)
