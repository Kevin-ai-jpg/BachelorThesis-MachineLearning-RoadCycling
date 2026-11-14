"""Decide whether a GoldenCheetah activity is a bike ride (decision D3).

280k activities in GoldenCheetah OpenData have no sport label, and many cycling ones carry a
label the audit did not know ("Rennrad", "Rolle", "Morning Ride"). Two steps:

1. `label_class`: map the free-text sport label to bike / nonbike / unknown with alias lists
   from the config (configs/data_sport_rule.yaml).
2. `rule_is_ride`: for unknown labels, accept an activity as a ride when it has power and its
   average cadence and speed look like cycling. Average speed separates the main confusion:
   running power meters (Stryd) also record power and a cadence of ~85 strides/min, but runners
   move at ~3 m/s while outdoor riders average ~7.5 m/s. Indoor rides have no (or zero) speed,
   so they are accepted by a separate "indoor" branch when allowed.

Only the activity-level metrics already in the index are used (no samples are read).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

BIKE, NONBIKE, UNKNOWN = "bike", "nonbike", "unknown"


def normalise_label(sport: pd.Series) -> pd.Series:
    return sport.fillna("").astype(str).str.strip().str.lower()


def label_class(sport: pd.Series, bike: list[str], nonbike: list[str]) -> pd.Series:
    s = normalise_label(sport)
    out = np.where(s.isin(bike), BIKE, np.where(s.isin(nonbike), NONBIKE, UNKNOWN))
    return pd.Series(out, index=sport.index)


def avg_speed_mps(rides: pd.DataFrame) -> pd.Series:
    """Distance / moving time; NaN when either is missing or moving time is 0."""
    t = rides["moving_time_s"].where(rides["moving_time_s"] > 0)
    return rides["distance_km"] * 1000.0 / t


@dataclass(frozen=True)
class Rule:
    speed_min_mps: float = 4.5
    speed_max_mps: float = 20.0
    allow_indoor: bool = True
    indoor_speed_mps: float = 0.5
    cadence_range_rpm: tuple[float, float] = (40.0, 130.0)
    min_avg_power_w: float = 1.0


def rule_is_ride(rides: pd.DataFrame, rule: Rule) -> pd.Series:
    """True where the activity looks like a ride with power (see module docstring)."""
    power = rides["has_power"].fillna(False).astype(bool) & (
        rides["avg_power_w"].fillna(0) >= rule.min_avg_power_w
    )
    cad = rides["avg_cadence_rpm"]
    lo, hi = rule.cadence_range_rpm
    cadence_ok = cad.isna() | cad.between(lo, hi)
    v = avg_speed_mps(rides)
    outdoor = v.between(rule.speed_min_mps, rule.speed_max_mps)
    indoor = (v.isna() | (v < rule.indoor_speed_mps)) if rule.allow_indoor else False
    return power & cadence_ok & (outdoor | indoor)


def has_power_baseline(rides: pd.DataFrame, min_avg_power_w: float = 1.0) -> pd.Series:
    """Naive baseline: every activity with power is a ride."""
    return rides["has_power"].fillna(False).astype(bool) & (
        rides["avg_power_w"].fillna(0) >= min_avg_power_w
    )


def tree_features(rides: pd.DataFrame) -> pd.DataFrame:
    """Inputs for the decision-tree comparison; missing values coded as -1."""
    v = avg_speed_mps(rides)
    km = rides["distance_km"].where(rides["distance_km"] > 0)
    return pd.DataFrame({
        "avg_power_w": rides["avg_power_w"],
        "avg_cadence_rpm": rides["avg_cadence_rpm"],
        "speed_mps": v,
        "climb_m_per_km": rides["elevation_gain_m"] / km,
        "duration_h": rides["duration_s"] / 3600,
        "has_gps": rides["has_gps"].astype(float),
        "has_hr": rides["has_hr"].astype(float),
        "np_over_avg": rides["np_w"] / rides["avg_power_w"].where(rides["avg_power_w"] > 0),
    }).fillna(-1.0)  # fmt: skip


def precision_recall(pred: pd.Series, truth: pd.Series) -> dict[str, float]:
    """truth: True = bike. Also the false-positive rate (share of non-rides accepted)."""
    pred, truth = pred.astype(bool), truth.astype(bool)
    tp = int((pred & truth).sum())
    fp = int((pred & ~truth).sum())
    fn = int((~pred & truth).sum())
    neg = int((~truth).sum())
    return {
        "precision": tp / (tp + fp) if tp + fp else float("nan"),
        "recall": tp / (tp + fn) if tp + fn else float("nan"),
        "fpr": fp / neg if neg else float("nan"),
        "tp": tp, "fp": fp, "fn": fn,
    }  # fmt: skip
