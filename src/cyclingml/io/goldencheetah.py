"""Read GoldenCheetah OpenData athlete zips.

Each zip holds one athlete (Liversedge 2020, doi:10.17605/OSF.IO/6HFPZ):

* `{<athlete-id>}.json` with
  - `ATHLETE`: gender, year of birth, id
  - `RIDES`: one entry per activity: `date` (UTC), `sport`, `data` (which channels were
    recorded, see CHANNEL_FLAGS) and `METRICS` (GoldenCheetah's own per-activity summary:
    TSS, NP, best power over 1 s to 90 min, the weight set in the athlete's profile, ...)
* one CSV per activity, named by its start time (`2017_07_18_12_01_20.csv`) with columns
  `secs,km,power,hr,cad,alt` (old devices may record every 5 s, not every second)

CSV file names use the device's local time while `date` is UTC, so they are matched on the
same minutes and seconds within +-15 h, as the official opendata-python library does.

    python -m cyclingml.io.goldencheetah index --zips <dir of zips> --out <dir> --workers 4

writes `athletes.parquet` and `rides/part-*.parquet` (one row per activity, no samples),
which the data audit (notebooks/02_data_audit.ipynb) reads.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import multiprocessing
import os
import re
import shutil
import zipfile
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Position -> letter in the RIDES[].data string, e.g. "TDSPHC-AGL-----".
CHANNEL_FLAGS = {
    "time": (0, "T"),
    "distance": (1, "D"),
    "speed": (2, "S"),
    "power": (3, "P"),
    "hr": (4, "H"),
    "cadence": (5, "C"),
    "torque": (6, "N"),
    "altitude": (7, "A"),
    "gps": (8, "G"),
}

# Best average power over these durations, as stored by GoldenCheetah ("<d>_critical_power").
MMP_DURATIONS = ["1s", "5s", "10s", "15s", "20s", "30s", "1m", "2m", "3m", "5m",
                 "8m", "10m", "20m", "30m", "60m", "90m"]  # fmt: skip

# GoldenCheetah metric name -> our column name.
METRIC_COLUMNS = {
    "workout_time": "duration_s",
    "time_riding": "moving_time_s",
    "total_distance": "distance_km",
    "elevation_gain": "elevation_gain_m",
    "average_power": "avg_power_w",
    "max_power": "max_power_w",
    "coggan_np": "np_w",
    "coggan_if": "if",
    "coggan_tss": "tss",
    "total_work": "work_kj",
    "average_hr": "avg_hr_bpm",
    "max_heartrate": "max_hr_bpm",
    "average_cad": "avg_cadence_rpm",
    "athlete_weight": "weight_kg",
    "cp_setting": "cp_setting_w",
    **{f"{d}_critical_power": f"mmp_{d}_w" for d in MMP_DURATIONS},
}

CSV_COLUMNS = {
    "secs": "time_s",
    "power": "power_w",
    "hr": "hr_bpm",
    "cad": "cadence_rpm",
    "alt": "altitude_m",
}
RIDE_SAMPLE_COLUMNS = ["time_s", "power_w", "hr_bpm", "cadence_rpm", "speed_mps",
                       "altitude_m", "distance_m"]  # fmt: skip

_SPAWN = multiprocessing.get_context("spawn")
_SENTINEL = -255.0  # GoldenCheetah's "not available"
_FILENAME_RE = re.compile(r"(\d{4})_(\d{2})_(\d{2})_(\d{2})_(\d{2})_(\d{2})\.csv$")


# --------------------------------------------------------------------------- metadata


def metric_value(raw: Any) -> float:
    """GoldenCheetah stores a metric as "12.3", ["12.3", "<count>"] or a list (histograms).

    Returns a float, NaN when missing, 'nan', the -255 sentinel, or a histogram list.
    """
    if isinstance(raw, list):
        if len(raw) == 2 and all(isinstance(x, str) for x in raw):
            raw = raw[0]
        else:
            return math.nan
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return math.nan
    return math.nan if (math.isnan(v) or v == _SENTINEL) else v


def parse_date(s: str) -> datetime:
    """'2017/07/18 11:01:20 UTC' -> aware UTC datetime."""
    return datetime.strptime(s.replace(" UTC", ""), "%Y/%m/%d %H:%M:%S").replace(
        tzinfo=timezone.utc
    )


def has_channel(flags: str, channel: str) -> bool:
    pos, letter = CHANNEL_FLAGS[channel]
    return len(flags) > pos and flags[pos] == letter


def _clean_id(raw: str) -> str:
    return raw.strip("{}").lower()


def read_metadata(zf: zipfile.ZipFile) -> dict:
    names = [n for n in zf.namelist() if n.lower().endswith(".json")]
    if not names:
        raise ValueError("no JSON metadata in zip")
    with zf.open(names[0]) as f:
        return json.load(f)


def match_csv(ride_utc: datetime, csv_times: dict[str, datetime]) -> str | None:
    """CSV whose (local-time) name has the same mm:ss and is within 15 h of the UTC start."""
    best, best_gap = None, timedelta(hours=15, minutes=1)
    for name, t in csv_times.items():
        if t.minute != ride_utc.minute or t.second != ride_utc.second:
            continue
        gap = abs(t.replace(tzinfo=timezone.utc) - ride_utc)
        if gap < best_gap:
            best, best_gap = name, gap
    return best


def _csv_times(names: Iterable[str]) -> dict[str, datetime]:
    out = {}
    for n in names:
        m = _FILENAME_RE.search(n)
        if m:
            out[n] = datetime(*map(int, m.groups()))
    return out


def ride_table(zip_path: str | Path) -> tuple[dict, pd.DataFrame]:
    """(athlete row, one row per activity) for one athlete zip. Reads metadata only."""
    zip_path = Path(zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        meta = read_metadata(zf)
        csvs = _csv_times(zf.namelist())
    ath = meta.get("ATHLETE", {})
    athlete_id = _clean_id(ath.get("id") or zip_path.stem)
    rows = []
    for ride in meta.get("RIDES", []):
        try:
            start = parse_date(ride["date"])
        except (KeyError, ValueError):
            continue
        flags = ride.get("data", "") or ""
        m = ride.get("METRICS", {}) or {}
        row = {
            "athlete_id": athlete_id,
            "start_utc": start,
            "sport": ride.get("sport", "") or "",
            "flags": flags,
            **{f"has_{c}": has_channel(flags, c) for c in CHANNEL_FLAGS},
            **{col: metric_value(m.get(key)) for key, col in METRIC_COLUMNS.items()},
            "csv_name": match_csv(start, csvs),
        }
        rows.append(row)
    rides = pd.DataFrame(rows)
    yob = metric_value(ath.get("yob"))
    athlete = {
        "athlete_id": athlete_id,
        "gender": (ath.get("gender") or "").upper()[:1] or None,
        "yob": yob,
        "zip_name": zip_path.name,
        "zip_mb": zip_path.stat().st_size / 1e6,
        "n_activities": len(rides),
        "n_csv": len(csvs),
    }
    return athlete, rides


# --------------------------------------------------------------------------- samples


def read_ride_csv(zip_path: str | Path, csv_name: str) -> pd.DataFrame:
    """One activity's samples in the project's standard columns (see CLAUDE.md).

    Not resampled: old files may have 5 s steps. Speed is derived from distance / time.
    """
    with zipfile.ZipFile(zip_path) as zf, zf.open(csv_name) as f:
        raw = pd.read_csv(f)
    df = pd.DataFrame({new: pd.to_numeric(raw[old], errors="coerce") if old in raw else np.nan
                       for old, new in CSV_COLUMNS.items()})  # fmt: skip
    km = pd.to_numeric(raw["km"], errors="coerce") if "km" in raw else pd.Series(np.nan, df.index)
    df["distance_m"] = km * 1000.0
    dt = df["time_s"].diff()
    df["speed_mps"] = (df["distance_m"].diff() / dt).where(dt > 0)
    if len(df):
        df.loc[df.index[0], "speed_mps"] = np.nan
    return df[RIDE_SAMPLE_COLUMNS]


# --------------------------------------------------------------------------- index


def _index_one(path: str) -> tuple[dict | None, pd.DataFrame | None, str | None]:
    if os.environ.get("CYCLINGML_TEST_CRASH_ON") == Path(path).name:  # test hook only
        os._exit(1)
    try:
        athlete, rides = ride_table(path)
        return athlete, rides, None
    except Exception as e:  # corrupt zip or JSON: record and move on
        return None, None, f"{type(e).__name__}: {e}"


def _read_progress(path: Path) -> dict[str, tuple[str, str]]:
    """zip name -> (status, detail) for every zip already indexed."""
    done: dict[str, tuple[str, str]] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            name, status, *detail = line.split("\t")
            done[name] = (status, detail[0] if detail else "")
    return done


def build_index(
    zip_dir: str | Path,
    out_dir: str | Path,
    workers: int = 2,
    batch: int = 200,
    rebuild: bool = False,
) -> dict[str, int]:
    """Index every athlete zip in zip_dir (metadata only, no samples).

    Writes out_dir/athletes.parquet and out_dir/rides/part-*.parquet.

    Resumable: zips already listed in out_dir/progress.tsv are skipped, so re-running after a
    disconnect continues where it stopped (rebuild=True starts over). Crash-safe: if a worker
    process dies (usually out of memory on a very large athlete file), the run continues with
    one worker, and a zip that still crashes is recorded as an error and skipped.
    """
    zip_dir, out_dir = Path(zip_dir), Path(out_dir)
    rides_dir, ath_dir = out_dir / "rides", out_dir / "athletes_parts"
    progress = out_dir / "progress.tsv"
    if rebuild:
        for d in (rides_dir, ath_dir):
            shutil.rmtree(d, ignore_errors=True)
        progress.unlink(missing_ok=True)
    rides_dir.mkdir(parents=True, exist_ok=True)
    ath_dir.mkdir(parents=True, exist_ok=True)

    done = _read_progress(progress)
    all_zips = sorted(zip_dir.glob("*.zip"))
    todo = [z for z in all_zips if z.name not in done]
    existing = [int(p.stem.split("-")[1]) for p in ath_dir.glob("part-*.parquet")]
    part = max(existing, default=-1) + 1
    rides_buf: list[pd.DataFrame] = []
    ath_buf: list[dict] = []
    new = 0

    def log_progress(lines: list[str]) -> None:
        with open(progress, "a", encoding="utf-8") as f:
            f.writelines(line + "\n" for line in lines)

    def flush() -> None:
        # parquet first, then progress: a crash in between only means re-indexing a batch
        nonlocal part, rides_buf, ath_buf
        if not ath_buf:
            return
        frames = [b for b in rides_buf if len(b)]
        if frames:
            pd.concat(frames, ignore_index=True).to_parquet(
                rides_dir / f"part-{part:05d}.parquet", index=False
            )
        pd.DataFrame(ath_buf).to_parquet(ath_dir / f"part-{part:05d}.parquet", index=False)
        log_progress([f"{a['zip_name']}\tok" for a in ath_buf])
        part += 1
        rides_buf, ath_buf = [], []

    def record_error(name: str, msg: str) -> None:
        log_progress([f"{name}\terror\t{msg.replace(chr(9), ' ').replace(chr(10), ' ')}"])

    remaining, w = list(todo), workers
    while remaining:
        handled: set[Path] = set()
        try:
            # spawn, not fork: forking a process that already runs threads can deadlock
            with ProcessPoolExecutor(max_workers=w, mp_context=_SPAWN) as pool:
                futures = {pool.submit(_index_one, str(z)): z for z in remaining}
                for fut in as_completed(futures):
                    z = futures[fut]
                    athlete, rides, err = fut.result()
                    handled.add(z)
                    new += 1
                    if err:
                        record_error(z.name, err)
                    else:
                        ath_buf.append(athlete)
                        rides_buf.append(rides)
                    if len(ath_buf) >= batch:
                        flush()
                    if new % 200 == 0:
                        log.info("indexed %d/%d new zips", new, len(todo))
            remaining = []
        except BrokenProcessPool:
            flush()
            pending = [z for z in remaining if z not in handled]
            if w > 1:
                log.warning("a worker crashed (out of memory?); continuing with 1 worker")
                w = 1
                remaining = pending
            else:
                crashed = pending[0]  # with one worker, tasks run in order
                log.warning("skipping %s: worker crashed while reading it", crashed.name)
                record_error(crashed.name, "worker crashed (probably out of memory)")
                new += 1
                remaining = pending[1:]
    flush()

    parts = sorted(ath_dir.glob("part-*.parquet"))
    athletes = (
        pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
        if parts
        else (pd.DataFrame())
    )
    athletes.to_parquet(out_dir / "athletes.parquet", index=False)
    status = _read_progress(progress)
    errors = [f"{n}\t{d}" for n, (st, d) in status.items() if st == "error"]
    (out_dir / "index_errors.txt").write_text("\n".join(errors), encoding="utf-8")
    return {
        "zips": len(all_zips),
        "new": new,
        "skipped": len(all_zips) - len(todo),
        "athletes": len(athletes),
        "errors": len(errors),
    }


def load_index(out_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    out_dir = Path(out_dir)
    athletes = pd.read_parquet(out_dir / "athletes.parquet")
    parts = sorted((out_dir / "rides").glob("part-*.parquet"))
    rides = (
        pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
        if parts
        else pd.DataFrame()
    )
    return athletes, rides


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Index GoldenCheetah OpenData zips")
    sub = p.add_subparsers(dest="cmd", required=True)
    pi = sub.add_parser("index")
    pi.add_argument("--zips", type=Path, required=True)
    pi.add_argument("--out", type=Path, required=True)
    pi.add_argument("--workers", type=int, default=2)
    pi.add_argument("--rebuild", action="store_true", help="start over instead of resuming")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    print(build_index(args.zips, args.out, args.workers, rebuild=args.rebuild))


if __name__ == "__main__":
    main()
