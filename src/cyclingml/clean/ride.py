"""Clean one activity into a 1 Hz frame (task 2.2, decision D4).

Input: the standard sample frame from a reader (`cyclingml.io.activity`,
`cyclingml.io.goldencheetah.read_ride_csv`), any sampling rate. Output: one row per second from
0 to the last sample, the standard columns plus `is_gap` (True where the device recorded nothing
for longer than `max_fill_gap_s`, e.g. auto-pause or a café stop).

Steps, in order (all thresholds come from the config, see configs/clean_rides.yaml):

1. **Reject the activity** if it is too short, longer than `max_duration_h` (broken clocks, a
   device left running), or if more than `max_impossible_speed_share` of samples are faster than
   `max_speed_mps` (car or train rides, GPS failure).
2. **Invalid samples -> NaN**: negative power, power above `max_power_w`, isolated power spikes,
   heart rate / cadence / speed outside physiological or physical limits.
   A *spike* is a sample more than `spike_delta_w` above the median of the centred window of
   `spike_window_s` samples. Real sprints last 5-20 s, so they move the median with them and
   survive; single-sample spikes from power-meter dropouts or interference do not.
   (The rolling-median "Hampel-type" filter: Pearson 2002, "Outliers in process modeling and
   identification", IEEE Trans. Control Systems Technology 10(1).)
3. **Resample to 1 Hz**: timestamps are rounded to whole seconds and averaged within a second.
4. **Fill short gaps** (<= `max_fill_gap_s` seconds of missing data, both missing seconds and
   NaN samples) by linear interpolation between the neighbouring samples. Longer gaps are not
   invented: a long *recording* gap (no samples at all) becomes power 0 / cadence 0 / speed 0 with
   `is_gap=True` (the rider was stopped, as GoldenCheetah does), a long *sensor* gap (samples
   present but the channel NaN) stays NaN.

Recorded zeros are kept: coasting (power 0 while moving) is real riding and lowers average power.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from cyclingml.io.goldencheetah import RIDE_SAMPLE_COLUMNS

CLEAN_COLUMNS = [*RIDE_SAMPLE_COLUMNS, "is_gap"]


@dataclass(frozen=True)
class CleanConfig:
    min_duration_s: float = 300.0
    max_duration_h: float = 12.0
    max_power_w: float = 2500.0
    spike_window_s: int = 7
    spike_delta_w: float = 700.0
    hr_range_bpm: tuple[float, float] = (30.0, 230.0)
    max_cadence_rpm: float = 200.0
    max_speed_mps: float = 33.0
    max_impossible_speed_share: float = 0.01
    max_fill_gap_s: int = 5

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> CleanConfig:
        d = dict(d or {})
        if "hr_range_bpm" in d:
            d["hr_range_bpm"] = tuple(d["hr_range_bpm"])
        unknown = set(d) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown clean settings: {sorted(unknown)}")
        return cls(**d)


@dataclass
class CleanResult:
    df: pd.DataFrame | None  # None when the activity is rejected
    reason: str | None = None  # why it was rejected
    report: dict[str, float] = field(default_factory=dict)  # counts of what was changed

    @property
    def kept(self) -> bool:
        return self.df is not None


# --------------------------------------------------------------------------- building blocks


def nan_runs(x: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) index pairs of every run of NaN in x."""
    isnan = np.isnan(x).astype(np.int8)
    edges = np.diff(np.concatenate([[0], isnan, [0]]))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True))


def fill_short_gaps(x: np.ndarray, max_len: int) -> np.ndarray:
    """Linearly interpolate NaN runs of length <= max_len that have a value on both sides.

    Runs at the start or end, and longer runs, stay NaN (pandas' `interpolate(limit=...)` would
    instead fill the first `limit` values of a long run, inventing data).
    """
    out = np.asarray(x, dtype=float).copy()
    n = len(out)
    for a, b in nan_runs(out):
        if b - a > max_len or a == 0 or b == n:
            continue
        left, right = out[a - 1], out[b]
        frac = np.arange(1, b - a + 1) / (b - a + 1)
        out[a:b] = left + frac * (right - left)
    return out


def spike_mask(power: np.ndarray, window: int, delta_w: float) -> np.ndarray:
    """True where power exceeds the centred rolling median by more than delta_w watts."""
    s = pd.Series(power)
    med = s.rolling(window, center=True, min_periods=1).median()
    return ((s - med) > delta_w).to_numpy()


def resample_1hz(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Average samples within each whole second onto the grid 0..last second.

    Returns the frame (seconds with no sample are NaN) and a mask of those empty seconds.
    """
    sec = np.round(df["time_s"].to_numpy(float)).astype(np.int64)
    sec -= sec.min()
    cols = RIDE_SAMPLE_COLUMNS[1:]
    grouped = df[cols].groupby(sec).mean()
    grid = np.arange(sec.max() + 1)
    out = grouped.reindex(grid)
    empty = ~np.isin(grid, grouped.index.to_numpy())
    out.insert(0, "time_s", grid.astype(float))
    return out.reset_index(drop=True), empty


# --------------------------------------------------------------------------- main entry


def clean_ride(df: pd.DataFrame, cfg: CleanConfig | None = None) -> CleanResult:
    """Clean one activity; see the module docstring for the rules."""
    cfg = cfg or CleanConfig()
    report: dict[str, float] = {"samples_in": float(len(df))}
    attrs = dict(df.attrs)
    df = df[RIDE_SAMPLE_COLUMNS].astype(float)
    df = df[df["time_s"].notna()].sort_values("time_s", kind="stable")
    if len(df) < 2:
        return CleanResult(None, "no samples", report)

    duration = df["time_s"].iloc[-1] - df["time_s"].iloc[0]
    report["duration_s"] = float(duration)
    if duration > cfg.max_duration_h * 3600:
        return CleanResult(None, f"longer than {cfg.max_duration_h:g} h", report)
    if duration < cfg.min_duration_s:
        return CleanResult(None, f"shorter than {cfg.min_duration_s:g} s", report)

    speed = df["speed_mps"].to_numpy()
    too_fast = speed > cfg.max_speed_mps
    share = too_fast.sum() / max(np.isfinite(speed).sum(), 1)
    report["impossible_speed_share"] = float(share)
    if share > cfg.max_impossible_speed_share:
        return CleanResult(
            None, f"{share:.1%} of samples faster than {cfg.max_speed_mps} m/s", report
        )

    df = df.copy()
    df.loc[too_fast, "speed_mps"] = np.nan
    df.loc[df["speed_mps"] < 0, "speed_mps"] = np.nan

    p = df["power_w"].to_numpy().copy()
    bad = (p < 0) | (p > cfg.max_power_w)
    report["power_out_of_range"] = float(bad.sum())
    p[bad] = np.nan
    spikes = spike_mask(p, cfg.spike_window_s, cfg.spike_delta_w)
    report["power_spikes"] = float(spikes.sum())
    p[spikes] = np.nan
    df["power_w"] = p

    lo, hi = cfg.hr_range_bpm
    hr = df["hr_bpm"]
    df.loc[(hr < lo) | (hr > hi), "hr_bpm"] = np.nan
    cad = df["cadence_rpm"]
    df.loc[(cad < 0) | (cad > cfg.max_cadence_rpm), "cadence_rpm"] = np.nan

    out, empty = resample_1hz(df)
    report["seconds"] = float(len(out))

    # long recording gaps (no samples): the rider was stopped
    is_gap = np.zeros(len(out), dtype=bool)
    for a, b in nan_runs(np.where(empty, np.nan, 0.0)):
        if b - a > cfg.max_fill_gap_s:
            is_gap[a:b] = True
    report["gap_seconds"] = float(is_gap.sum())

    for col in RIDE_SAMPLE_COLUMNS[1:]:
        out[col] = fill_short_gaps(out[col].to_numpy(), cfg.max_fill_gap_s)
    for col in ("power_w", "cadence_rpm", "speed_mps"):
        v = out[col].to_numpy().copy()
        v[is_gap] = 0.0
        out[col] = v
    # distance does not move while stopped: carry it forward across the gap
    out["distance_m"] = out["distance_m"].ffill().where(out["distance_m"].notna() | is_gap)
    out["is_gap"] = is_gap

    report["power_nan_seconds"] = float(out["power_w"].isna().sum())
    out = out[CLEAN_COLUMNS]
    out.attrs = attrs
    return CleanResult(out, None, report)
