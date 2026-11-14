"""Read one activity file (FIT, GPX, TCX, optionally gzipped) into the standard sample frame.

Output columns are the project standard (CLAUDE.md), the same as
`cyclingml.io.goldencheetah.read_ride_csv`:

    time_s, power_w, hr_bpm, cadence_rpm, speed_mps, altitude_m, distance_m

`time_s` counts seconds from the first sample. The frame is *not* resampled (devices may record
every 1-5 s or only on change, "smart recording"); `cyclingml.clean.ride` resamples to 1 Hz.
GPS positions are read only to derive distance when the file has none, and are then dropped:
the frame never contains coordinates (privacy of the amateur riders, task 1.5).

The start time (UTC) is stored in `df.attrs["start_utc"]`, the source format in
`df.attrs["format"]`.

Formats:
* FIT: Garmin's binary format, read with `fitdecode`; one sample per `record` message.
* GPX: GPS track; heart rate, cadence and power come from the Garmin TrackPointExtension and the
  `<power>` extension that Garmin/Strava exports write. Distance and speed are computed from the
  coordinates (haversine).
* TCX: Garmin Training Center XML; speed and power from the ActivityExtension `TPX` element.
"""

from __future__ import annotations

import gzip
import io
import zipfile
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd

from cyclingml.io.goldencheetah import RIDE_SAMPLE_COLUMNS

SUPPORTED_SUFFIXES = (".fit", ".gpx", ".tcx")
EARTH_RADIUS_M = 6_371_000.0


class UnsupportedActivity(ValueError):
    """File is not a FIT/GPX/TCX activity, or holds no samples."""


# --------------------------------------------------------------------------- helpers


def activity_format(name: str) -> str | None:
    """'ride.fit.gz' -> 'fit'; None for unsupported names."""
    n = name.lower()
    if n.endswith(".gz"):
        n = n[:-3]
    for suf in SUPPORTED_SUFFIXES:
        if n.endswith(suf):
            return suf[1:]
    return None


def haversine_m(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Distance in metres between consecutive points (first element 0).

    Great-circle distance on a sphere of radius 6,371 km (haversine formula, Sinnott 1984,
    "Virtues of the Haversine", Sky and Telescope 68(2)). Error vs the WGS-84 ellipsoid is
    below 0.5%, far smaller than GPS noise. NaN coordinates give NaN steps.
    """
    la, lo = np.radians(lat), np.radians(lon)
    dlat, dlon = np.diff(la), np.diff(lo)
    a = np.sin(dlat / 2) ** 2 + np.cos(la[:-1]) * np.cos(la[1:]) * np.sin(dlon / 2) ** 2
    d = 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    return np.concatenate([[0.0], d])


def _to_utc(t: datetime) -> datetime:
    return t.replace(tzinfo=timezone.utc) if t.tzinfo is None else t.astimezone(timezone.utc)


def _frame(rows: list[dict[str, Any]], fmt: str) -> pd.DataFrame:
    """Rows with a 'time' (aware datetime) and any standard columns -> standard frame."""
    rows = [r for r in rows if r.get("time") is not None]
    if not rows:
        raise UnsupportedActivity(f"{fmt}: no timestamped samples")
    raw = pd.DataFrame(rows)
    t = pd.to_datetime(raw["time"], utc=True)
    start = t.min()
    df = pd.DataFrame({"time_s": (t - start).dt.total_seconds().to_numpy()})
    for col in RIDE_SAMPLE_COLUMNS[1:]:
        df[col] = pd.to_numeric(raw[col], errors="coerce") if col in raw else np.nan
    # distance from GPS when the file has none
    if df["distance_m"].isna().all() and {"lat", "lon"} <= set(raw.columns):
        lat = pd.to_numeric(raw["lat"], errors="coerce").to_numpy(float)
        lon = pd.to_numeric(raw["lon"], errors="coerce").to_numpy(float)
        if np.isfinite(lat).sum() >= 2:
            step = np.nan_to_num(haversine_m(lat, lon), nan=0.0)
            df["distance_m"] = np.cumsum(step)
    df = df.sort_values("time_s", kind="stable").reset_index(drop=True)
    # speed from distance when the file has none
    if df["speed_mps"].isna().all() and df["distance_m"].notna().any():
        dt = df["time_s"].diff()
        df["speed_mps"] = (df["distance_m"].diff() / dt).where(dt > 0)
    df = df[RIDE_SAMPLE_COLUMNS].astype(float)
    df.attrs["start_utc"] = start.to_pydatetime()
    df.attrs["format"] = fmt
    return df


# --------------------------------------------------------------------------- FIT

# record message field -> our column; "enhanced_*" (32-bit) wins over the 16-bit field
_FIT_FIELDS = {
    "power": "power_w",
    "heart_rate": "hr_bpm",
    "cadence": "cadence_rpm",
    "speed": "speed_mps",
    "enhanced_speed": "speed_mps",
    "altitude": "altitude_m",
    "enhanced_altitude": "altitude_m",
    "distance": "distance_m",
}


def _semicircles_to_deg(v: Any) -> float | None:
    """FIT stores positions as int32 'semicircles': degrees = v * 180 / 2^31."""
    return None if v is None else float(v) * 180.0 / 2**31


def read_fit(data: bytes) -> pd.DataFrame:
    import fitdecode  # imported here so GPX/TCX users do not need it at import time

    rows: list[dict[str, Any]] = []
    with fitdecode.FitReader(io.BytesIO(data)) as fit:
        for frame in fit:
            if frame.frame_type != fitdecode.FIT_FRAME_DATA or frame.name != "record":
                continue
            row: dict[str, Any] = {}
            enhanced: set[str] = set()
            for field in frame.fields:
                name, value = field.name, field.value
                if name == "timestamp" and isinstance(value, datetime):
                    row["time"] = _to_utc(value)
                elif name in ("position_lat", "position_long") and value is not None:
                    row["lat" if name.endswith("lat") else "lon"] = _semicircles_to_deg(value)
                elif name in _FIT_FIELDS and value is not None:
                    col = _FIT_FIELDS[name]
                    if name.startswith("enhanced_"):
                        enhanced.add(col)
                    elif col in enhanced:
                        continue
                    try:
                        row[col] = float(value)
                    except (TypeError, ValueError):
                        continue
            rows.append(row)
    return _frame(rows, "fit")


# --------------------------------------------------------------------------- GPX


def _local(tag: str) -> str:
    """'{namespace}hr' -> 'hr'."""
    return tag.rsplit("}", 1)[-1].lower()


# extension element (local name, lower case) -> our column
_GPX_EXT = {
    "hr": "hr_bpm",
    "heartrate": "hr_bpm",
    "cad": "cadence_rpm",
    "cadence": "cadence_rpm",
    "power": "power_w",
    "watts": "power_w",
    "speed": "speed_mps",
}


def read_gpx(data: bytes) -> pd.DataFrame:
    import gpxpy

    gpx = gpxpy.parse(data.decode("utf-8-sig", errors="replace"))
    rows: list[dict[str, Any]] = []
    for track in gpx.tracks:
        for seg in track.segments:
            for p in seg.points:
                row: dict[str, Any] = {
                    "time": _to_utc(p.time) if p.time else None,
                    "lat": p.latitude,
                    "lon": p.longitude,
                    "altitude_m": p.elevation,
                }
                for ext in p.extensions:
                    for el in ext.iter():
                        col = _GPX_EXT.get(_local(el.tag))
                        if col and el.text and el.text.strip():
                            row[col] = el.text.strip()
                rows.append(row)
    return _frame(rows, "gpx")


# --------------------------------------------------------------------------- TCX


def _child(el: ET.Element, name: str) -> ET.Element | None:
    for c in el:
        if _local(c.tag) == name:
            return c
    return None


def _text(el: ET.Element | None) -> str | None:
    return el.text.strip() if el is not None and el.text else None


def read_tcx(data: bytes) -> pd.DataFrame:
    root = ET.fromstring(data.lstrip())  # some exporters write blank lines before <?xml
    rows: list[dict[str, Any]] = []
    for tp in root.iter():
        if _local(tp.tag) != "trackpoint":
            continue
        t = _text(_child(tp, "time"))
        row: dict[str, Any] = {
            "time": _to_utc(datetime.fromisoformat(t.replace("Z", "+00:00"))) if t else None,
            "altitude_m": _text(_child(tp, "altitudemeters")),
            "distance_m": _text(_child(tp, "distancemeters")),
            "cadence_rpm": _text(_child(tp, "cadence")),
        }
        hr = _child(tp, "heartratebpm")
        if hr is not None:
            row["hr_bpm"] = _text(_child(hr, "value"))
        pos = _child(tp, "position")
        if pos is not None:
            row["lat"] = _text(_child(pos, "latitudedegrees"))
            row["lon"] = _text(_child(pos, "longitudedegrees"))
        ext = _child(tp, "extensions")
        if ext is not None:
            for el in ext.iter():
                name = _local(el.tag)
                if name == "watts":
                    row["power_w"] = _text(el)
                elif name == "speed":
                    row["speed_mps"] = _text(el)
        rows.append(row)
    return _frame(rows, "tcx")


# --------------------------------------------------------------------------- dispatch

_READERS = {"fit": read_fit, "gpx": read_gpx, "tcx": read_tcx}


def read_activity_bytes(data: bytes, name: str) -> pd.DataFrame:
    """Read an activity from memory; the format comes from the name ('x.fit.gz', 'y.gpx')."""
    fmt = activity_format(name)
    if fmt is None:
        raise UnsupportedActivity(f"unsupported file type: {name}")
    if name.lower().endswith(".gz") or data[:2] == b"\x1f\x8b":
        data = gzip.decompress(data)
    try:
        df = _READERS[fmt](data)
    except UnsupportedActivity:
        raise
    except Exception as e:  # corrupt file: one clear error type for callers
        raise UnsupportedActivity(f"{name}: {type(e).__name__}: {e}") from e
    df.attrs["source"] = name
    return df


def read_activity(path: str | Path) -> pd.DataFrame:
    """Read a FIT/GPX/TCX file (optionally .gz) into the standard sample frame."""
    path = Path(path)
    return read_activity_bytes(path.read_bytes(), path.name)


def iter_activity_files(root: str | Path) -> Iterator[tuple[str, bytes]]:
    """(name, bytes) for every activity file under root, including inside .zip files.

    Handles the usual exports: a Strava archive (`activities/*.fit.gz|*.gpx|*.tcx.gz`) and a
    Garmin bulk export (FIT files inside nested `UploadedFiles_*.zip`). Order is sorted by name.
    """
    root = Path(root)
    paths = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file())
    for p in paths:
        if p.suffix.lower() == ".zip":
            yield from _iter_zip(zipfile.ZipFile(p), prefix=p.name)
        elif activity_format(p.name):
            yield (p.relative_to(root).as_posix() if p != root else p.name), p.read_bytes()


def _iter_zip(zf: zipfile.ZipFile, prefix: str) -> Iterator[tuple[str, bytes]]:
    with zf:
        for info in sorted(zf.infolist(), key=lambda i: i.filename):
            if info.is_dir():
                continue
            name = f"{prefix}/{info.filename}"
            if info.filename.lower().endswith(".zip"):
                yield from _iter_zip(zipfile.ZipFile(io.BytesIO(zf.read(info))), name)
            elif activity_format(info.filename):
                yield name, zf.read(info)
