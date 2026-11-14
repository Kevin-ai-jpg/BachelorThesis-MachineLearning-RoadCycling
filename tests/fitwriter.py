"""Minimal FIT file writer, only for tests: builds activity files with known samples.

Follows the FIT protocol (Garmin FIT SDK, "Flexible and Interoperable Data Transfer"):
14-byte header, definition + data messages, CRC-16 over the whole file.
Only the `file_id` (global 0) and `record` (global 20) messages are written.
"""

from __future__ import annotations

import struct
from datetime import datetime, timezone

FIT_EPOCH = datetime(1989, 12, 31, tzinfo=timezone.utc)
_CRC_TABLE = [0x0000, 0xCC01, 0xD801, 0x1400, 0xF001, 0x3C00, 0x2800, 0xE401,
              0xA001, 0x6C00, 0x7800, 0xB401, 0x5000, 0x9C01, 0x8801, 0x4400]  # fmt: skip

# record field: (field number, struct code, FIT base type, scale, offset, invalid value)
_RECORD_FIELDS = {
    "timestamp": (253, "I", 0x86, 1, 0, 0xFFFFFFFF),
    "position_lat": (0, "i", 0x85, 1, 0, 0x7FFFFFFF),
    "position_long": (1, "i", 0x85, 1, 0, 0x7FFFFFFF),
    "heart_rate": (3, "B", 0x02, 1, 0, 0xFF),
    "cadence": (4, "B", 0x02, 1, 0, 0xFF),
    "distance": (5, "I", 0x86, 100, 0, 0xFFFFFFFF),
    "power": (7, "H", 0x84, 1, 0, 0xFFFF),
    "enhanced_speed": (73, "I", 0x86, 1000, 0, 0xFFFFFFFF),
    "enhanced_altitude": (78, "I", 0x86, 5, 500, 0xFFFFFFFF),
}
_SIZES = {"B": 1, "H": 2, "I": 4, "i": 4}


def fit_crc(data: bytes, crc: int = 0) -> int:
    for byte in data:
        tmp = _CRC_TABLE[crc & 0xF]
        crc = (crc >> 4) & 0x0FFF
        crc = crc ^ tmp ^ _CRC_TABLE[byte & 0xF]
        tmp = _CRC_TABLE[crc & 0xF]
        crc = (crc >> 4) & 0x0FFF
        crc = crc ^ tmp ^ _CRC_TABLE[(byte >> 4) & 0xF]
    return crc


def _definition(local: int, global_num: int, fields: list[tuple[int, str, int]]) -> bytes:
    out = struct.pack("<BBBHB", 0x40 | local, 0, 0, global_num, len(fields))
    for num, code, base in fields:
        out += struct.pack("<BBB", num, _SIZES[code], base)
    return out


def write_fit(records: list[dict]) -> bytes:
    """records: dicts with 'time' (aware datetime) and any of power, heart_rate, cadence,
    distance (m), enhanced_speed (m/s), enhanced_altitude (m), lat, lon (degrees).
    A missing or None value is written as the FIT 'invalid' value."""
    body = _definition(0, 0, [(0, "B", 0x00)]) + struct.pack("<BB", 0, 4)  # file_id: activity
    names = list(_RECORD_FIELDS)
    body += _definition(1, 20, [(_RECORD_FIELDS[n][0], _RECORD_FIELDS[n][1],
                                 _RECORD_FIELDS[n][2]) for n in names])  # fmt: skip
    for r in records:
        values = dict(r)
        values["timestamp"] = int((r["time"] - FIT_EPOCH).total_seconds())
        if r.get("lat") is not None:
            values["position_lat"] = round(r["lat"] * 2**31 / 180)
            values["position_long"] = round(r["lon"] * 2**31 / 180)
        msg = struct.pack("<B", 1)
        for n in names:
            _, code, _, scale, offset, invalid = _RECORD_FIELDS[n]
            v = values.get(n)
            raw = invalid if v is None else round((v + offset) * scale)
            msg += struct.pack("<" + code, raw)
        body += msg
    header = struct.pack("<BBHI4s", 14, 0x20, 2132, len(body), b".FIT")
    header += struct.pack("<H", fit_crc(header))
    data = header + body
    return data + struct.pack("<H", fit_crc(data))
