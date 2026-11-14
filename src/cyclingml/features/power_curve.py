"""Power curve (mean-maximal power, MMP) per activity and per rider (task 2.3).

The power curve of an activity gives, for each duration d, the highest average power the rider
held over any d consecutive seconds. Across many activities the upper envelope is the rider's
"record power profile" (Pinot & Grappe 2011, "The record power profile to assess performance in
elite cyclists", Int J Sports Med 32(11):839-844), the basis of rider-type profiling and of the
comparison with pro riders (Leo et al. 2022).

**Fast method.** With the prefix sums C[i] = p[0] + ... + p[i-1] (C[0] = 0), the sum over the
window starting at i is C[i+d] - C[i], so every window of length d costs one subtraction and one
duration costs O(n) instead of the brute-force O(n*d). For a 3 h ride (n = 10,800 s) and ~30
durations this is ~300k operations rather than ~10^8.

Input must be the cleaned 1 Hz frame (`cyclingml.clean.ride.clean_ride`): one sample per second,
so a window of d samples is d seconds. Missing power (NaN, a long sensor dropout) counts as
0 W: we never claim a power the rider did not demonstrably produce.

Note: the curve is *not* guaranteed to decrease with duration. For [10, 0, 10] W the best 2 s is
5 W but the best 3 s is 6.7 W. What always holds is best(k*d) <= best(d) for whole k (a k*d
window splits into k windows of d). On real rides the curve is decreasing in practice.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

# 1 s to 3 h. Dense where the curve bends (sprint and VO2max range), sparse in the flat tail.
DEFAULT_DURATIONS_S: tuple[int, ...] = (
    1, 2, 3, 5, 10, 15, 20, 30, 45, 60, 90, 120, 180, 240, 300, 360, 480, 600, 720,
    900, 1200, 1800, 2400, 3600, 5400, 7200, 10800,
)  # fmt: skip


def column_name(d: int) -> str:
    return f"mmp_{d}s"


def best_powers(power: np.ndarray, durations: Sequence[int] = DEFAULT_DURATIONS_S) -> np.ndarray:
    """Best average power (W) over every duration (s) in `durations`; NaN if the activity is
    shorter than the duration. `power` is 1 Hz; NaN counts as 0 W (see module docstring)."""
    p = np.nan_to_num(np.asarray(power, dtype=float), nan=0.0)
    n = len(p)
    c = np.concatenate([[0.0], np.cumsum(p)])
    out = np.full(len(durations), np.nan)
    for k, d in enumerate(durations):
        if d < 1:
            raise ValueError(f"duration must be >= 1 s, got {d}")
        if d <= n:
            out[k] = float(np.max(c[d:] - c[:-d])) / d
    return out


def best_powers_bruteforce(power: np.ndarray, durations: Sequence[int]) -> np.ndarray:
    """Reference implementation (every window summed directly). Tests only: O(n*d)."""
    p = np.nan_to_num(np.asarray(power, dtype=float), nan=0.0)
    out = np.full(len(durations), np.nan)
    for k, d in enumerate(durations):
        if d <= len(p):
            out[k] = max(p[i : i + d].sum() for i in range(len(p) - d + 1)) / d
    return out


def activity_curve(
    ride: pd.DataFrame, durations: Sequence[int] = DEFAULT_DURATIONS_S
) -> dict[str, float]:
    """{'mmp_1s': ..., 'mmp_5s': ...} for one cleaned 1 Hz activity."""
    vals = best_powers(ride["power_w"].to_numpy(), durations)
    return {column_name(d): float(v) for d, v in zip(durations, vals, strict=True)}


def curve_columns(curves: pd.DataFrame) -> list[str]:
    cols = [c for c in curves.columns if c.startswith("mmp_") and c.endswith("s")]
    return sorted(cols, key=lambda c: int(c[4:-1]))


def rolling_best(
    curves: pd.DataFrame,
    window_days: int = 90,
    include_current: bool = True,
    rider_col: str = "rider_id",
    time_col: str = "start_utc",
) -> pd.DataFrame:
    """Each rider's best power per duration over the `window_days` before each activity.

    `curves` has one row per activity (rider, start time, mmp_* columns). Returns the same rows
    and order with the mmp_* columns replaced by the rolling best.

    include_current=True: window (t - window, t], the curve *after* this ride (describes form).
    include_current=False: window [t - window, t), only earlier rides, the version to use as a
    model *input* when predicting something about the ride at t (no future/same-ride leakage).
    A rider's first activity then has NaN.
    """
    cols = curve_columns(curves)
    closed = "right" if include_current else "left"
    out = curves.copy()
    order = curves.sort_values([rider_col, time_col], kind="stable").index
    parts = []
    for _, g in curves.loc[order].groupby(rider_col, sort=False):
        ts = pd.to_datetime(g[time_col], utc=True)
        rolled = g[cols].set_axis(ts).rolling(f"{window_days}D", closed=closed).max()
        parts.append(rolled.set_axis(g.index))
    if parts:
        out[cols] = pd.concat(parts).loc[curves.index, cols]
    return out


def alltime_best(curves: pd.DataFrame, rider_col: str = "rider_id") -> pd.DataFrame:
    """One row per rider: best power per duration over all activities."""
    return curves.groupby(rider_col)[curve_columns(curves)].max()
