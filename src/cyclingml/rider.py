"""Process one rider's activity history: read -> clean (1 Hz) -> power curves -> summary.

Tasks 2.1-2.3 end to end. Input is either an amateur export folder (Strava archive or Garmin
bulk export, unzipped or not) or a GoldenCheetah OpenData athlete zip:

    python -m cyclingml.rider <export folder> --rider-id rider01 [--weight-kg 72]
    python -m cyclingml.rider <athlete.zip> --goldencheetah --rider-id gc_<id>

Writes to $CYCLINGML_DATA/processed/riders/<rider-id>/ (or --out):
    activities.parquet   one row per file: start time, kept / rejection reason, cleaning counts,
                         power curve (mmp_<d>s), rolling 90-day best (best90_mmp_<d>s)
    rides/<start>.parquet the cleaned 1 Hz samples of every kept activity (with --save-rides)
    summary.md           counts, date range, all-time and last-90-day bests
"""

from __future__ import annotations

import argparse
import logging
import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cyclingml import paths
from cyclingml.clean.ride import CleanConfig, clean_ride
from cyclingml.config import load_config
from cyclingml.features.power_curve import (
    DEFAULT_DURATIONS_S,
    activity_curve,
    column_name,
    curve_columns,
    rolling_best,
)
from cyclingml.io import goldencheetah as gc
from cyclingml.io.activity import UnsupportedActivity, iter_activity_files, read_activity_bytes

log = logging.getLogger(__name__)
DEFAULT_CONFIG = paths.REPO_ROOT / "configs" / "clean_rides.yaml"
SUMMARY_DURATIONS = (5, 60, 300, 1200, 3600)
DUPLICATE_TOLERANCE_S = 120  # same start within 2 min = same ride exported twice


def _label(d: int) -> str:
    return f"{d} s" if d < 60 else f"{d // 60} min"


def iter_amateur(root: Path) -> Iterator[tuple[str, pd.DataFrame | None, str | None]]:
    """(source name, raw samples or None, error) for every activity file in an export."""
    for name, data in iter_activity_files(root):
        try:
            yield name, read_activity_bytes(data, name), None
        except UnsupportedActivity as e:
            yield name, None, str(e)


def iter_goldencheetah(zip_path: Path) -> Iterator[tuple[str, pd.DataFrame | None, str | None]]:
    """Cycling activities of one GoldenCheetah athlete zip, with GC's own metrics in attrs."""
    _, rides = gc.ride_table(zip_path)
    bike = rides[(rides["sport"] == "Bike") & rides["csv_name"].notna()]
    for _, r in bike.iterrows():
        try:
            df = gc.read_ride_csv(zip_path, r["csv_name"])
        except (KeyError, ValueError, zipfile.BadZipFile) as e:
            yield r["csv_name"], None, f"{type(e).__name__}: {e}"
            continue
        df.attrs["start_utc"] = r["start_utc"].to_pydatetime()
        df.attrs["format"] = "goldencheetah"
        df.attrs["gc_metrics"] = {c: r[c] for c in rides.columns if c.startswith("mmp_")}
        yield r["csv_name"], df, None


def process_rider(
    source: Iterator[tuple[str, pd.DataFrame | None, str | None]],
    rider_id: str,
    cfg: dict[str, Any],
    rides_dir: Path | None = None,
    durations: tuple[int, ...] = DEFAULT_DURATIONS_S,
) -> pd.DataFrame:
    """One row per activity file: status, cleaning report and power curve."""
    clean_cfg = CleanConfig.from_dict(cfg.get("clean"))
    min_power_share = float(cfg.get("power_curve", {}).get("min_power_share", 0.5))
    rows: list[dict[str, Any]] = []
    for name, raw, err in source:
        row: dict[str, Any] = {"rider_id": rider_id, "source": name}
        if raw is None:
            rows.append({**row, "status": "unreadable", "reason": err})
            continue
        row["start_utc"] = pd.Timestamp(raw.attrs.get("start_utc"))
        row["format"] = raw.attrs.get("format")
        row.update({f"gc_{k}": v for k, v in raw.attrs.get("gc_metrics", {}).items()})
        res = clean_ride(raw, clean_cfg)
        row.update(res.report)
        if not res.kept:
            rows.append({**row, "status": "rejected", "reason": res.reason})
            continue
        ride = res.df
        riding = ~ride["is_gap"].to_numpy()
        share = float(ride["power_w"][riding].notna().mean()) if riding.any() else 0.0
        has_power = share >= min_power_share and float(ride["power_w"].max()) > 0
        row.update({"status": "kept", "power_share": share, "has_power": has_power,
                    "moving_s": float(riding.sum())})  # fmt: skip
        if has_power:
            row.update(activity_curve(ride, durations))
            row["avg_power_w"] = float(ride["power_w"][riding].mean())
        if rides_dir is not None:
            rides_dir.mkdir(parents=True, exist_ok=True)
            # numbered: two files of the same ride may start in the same second
            path = rides_dir / f"{row['start_utc']:%Y%m%dT%H%M%S}_{len(rows):05d}.parquet"
            ride = ride.copy()
            ride.attrs = {}  # pandas would store attrs as JSON; datetimes are not JSON
            ride.to_parquet(path, index=False)
            row["ride_file"] = path.name
        rows.append(row)
    acts = pd.DataFrame(rows)
    if acts.empty:
        return acts
    for col in ("has_power", "power_share", "moving_s", "start_utc"):  # even if nothing kept
        if col not in acts:
            acts[col] = pd.NaT if col == "start_utc" else np.nan
    acts = mark_duplicates(acts)
    if rides_dir is not None and "ride_file" in acts:
        for name in acts.loc[acts["status"] == "duplicate", "ride_file"].dropna():
            (rides_dir / name).unlink(missing_ok=True)
        acts.loc[acts["status"] == "duplicate", "ride_file"] = None
    for d in durations:  # same columns even if no activity had power
        if column_name(d) not in acts:
            acts[column_name(d)] = np.nan
    power = acts[(acts["status"] == "kept") & acts["has_power"].eq(True)]
    if len(power):
        window = int(cfg.get("power_curve", {}).get("rolling_window_days", 90))
        rolled = rolling_best(power, window_days=window)
        for c in curve_columns(power):
            acts.loc[power.index, f"best{window}_{c}"] = rolled[c]
    return acts.sort_values("start_utc", na_position="last").reset_index(drop=True)


def mark_duplicates(acts: pd.DataFrame) -> pd.DataFrame:
    """Same ride in two files (e.g. .fit and .gpx, or Garmin + Strava exports): keep the one
    with the most seconds of power, mark the others status='duplicate'."""
    kept = acts[acts["status"] == "kept"].copy()
    if kept.empty:
        return acts
    kept["_power_s"] = kept["power_share"].fillna(0) * kept["moving_s"].fillna(0)
    kept = kept.sort_values("start_utc")
    group = (kept["start_utc"].diff().dt.total_seconds() > DUPLICATE_TOLERANCE_S).cumsum()
    best = kept.groupby(group.to_numpy())["_power_s"].idxmax()
    dup = kept.index.difference(best.to_numpy())
    acts.loc[dup, "status"] = "duplicate"
    return acts


def summary_markdown(acts: pd.DataFrame, rider_id: str, weight_kg: float | None = None) -> str:
    status = acts["status"].value_counts()
    lines = [f"# Rider {rider_id}", "", "## Files", ""]
    lines += [f"- {k}: {v}" for k, v in status.items()]
    rej = acts.loc[acts["status"].isin(["rejected", "unreadable"]), "reason"].value_counts()
    if len(rej):
        lines += ["", "Rejected / unreadable, by reason:", ""]
        lines += [f"- {k}: {v}" for k, v in rej.head(10).items()]
    kept = acts[acts["status"] == "kept"]
    power = kept[kept["has_power"].eq(True)]
    if kept.empty:
        return "\n".join(lines) + "\n"
    first, last = kept["start_utc"].min(), kept["start_utc"].max()
    weeks = max((last - first).days / 7, 1)
    lines += ["", "## History", "",
              f"- Activities kept: {len(kept)}, with power: {len(power)}",
              f"- From {first:%Y-%m-%d} to {last:%Y-%m-%d} ({(last - first).days} days)",
              f"- Power rides per week: {len(power) / weeks:.1f}",
              f"- Riding time: {kept['moving_s'].sum() / 3600:.0f} h"]  # fmt: skip
    cleaned = {k: kept[k].sum() for k in ("power_spikes", "power_out_of_range") if k in kept}
    lines += [f"- Power samples removed: {int(cleaned.get('power_spikes', 0))} spikes, "
              f"{int(cleaned.get('power_out_of_range', 0))} out of range"]  # fmt: skip
    if len(power):
        recent = power[power["start_utc"] > last - pd.Timedelta(days=90)]
        lines += ["", "## Best power (cleaned data)", ""]
        head = "| Duration | All-time W | Last 90 days W |"
        sep = "| --- | --- | --- |"
        if weight_kg:
            head += " All-time W/kg | Last 90 days W/kg |"
            sep += " --- | --- |"
        lines += [head, sep]
        for d in SUMMARY_DURATIONS:
            c = column_name(d)
            a, r = power[c].max(), recent[c].max() if len(recent) else np.nan
            row = f"| {_label(d)} | {a:.0f} | {r:.0f} |"
            if weight_kg:
                row += f" {a / weight_kg:.2f} | {r / weight_kg:.2f} |"
            lines.append(row)
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Clean a rider's activities and compute power curves")
    p.add_argument("source", type=Path, help="export folder/zip, or a GoldenCheetah athlete zip")
    p.add_argument("--rider-id", required=True, help="pseudonym, never the rider's name")
    p.add_argument("--goldencheetah", action="store_true", help="source is a GC OpenData zip")
    p.add_argument("--weight-kg", type=float, default=None)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--save-rides", action="store_true", help="also write cleaned 1 Hz rides")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = load_config(args.config)
    out = args.out or paths.processed_dir() / "riders" / args.rider_id
    out.mkdir(parents=True, exist_ok=True)
    src = iter_goldencheetah(args.source) if args.goldencheetah else iter_amateur(args.source)
    acts = process_rider(src, args.rider_id, cfg, out / "rides" if args.save_rides else None)
    acts.to_parquet(out / "activities.parquet", index=False)
    md = summary_markdown(acts, args.rider_id, args.weight_kg)
    (out / "summary.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"written to {out}")


if __name__ == "__main__":
    main()
