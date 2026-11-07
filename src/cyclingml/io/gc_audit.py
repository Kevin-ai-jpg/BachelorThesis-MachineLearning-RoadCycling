"""Data audit of the GoldenCheetah OpenData index (task 1.2).

Answers the questions that decide how the thesis uses this data:

1. How many riders and cycling activities, and how many have a power meter?
2. Is body weight usable for W/kg, or mostly missing / left at a default value?
3. How many riders have enough power history for profiling and for training-response models?
4. Are there implausible values that cleaning must handle?
5. Gate 1 from the plan: at least `gate_min_riders` riders with usable power history.

Thresholds come from configs/audit_goldencheetah.yaml.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

CYCLING_SPORTS = {"bike", "biking", "cycling", "ride", "turbo", "virtualride", "mtb", "road"}


@dataclass
class AuditConfig:
    min_power_rides: int = 100  # bike activities with power, per rider
    min_span_days: int = 180  # first to last power ride
    weight_range_kg: tuple[float, float] = (40.0, 130.0)
    suspect_default_weights: tuple[float, ...] = (75.0,)  # values to check as possible defaults
    max_plausible_w: dict[str, float] = field(
        default_factory=lambda: {"mmp_5s_w": 2500.0, "mmp_1m_w": 1200.0,
                                 "mmp_5m_w": 650.0, "mmp_20m_w": 550.0}
    )  # fmt: skip
    gate_min_riders: int = 500

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AuditConfig:
        d = dict(d)
        if "weight_range_kg" in d:
            d["weight_range_kg"] = tuple(d["weight_range_kg"])
        if "suspect_default_weights" in d:
            d["suspect_default_weights"] = tuple(d["suspect_default_weights"])
        return cls(**d)


def is_cycling(sport: pd.Series) -> pd.Series:
    return sport.fillna("").str.strip().str.lower().isin(CYCLING_SPORTS)


def power_rides(rides: pd.DataFrame) -> pd.DataFrame:
    """Cycling activities that recorded power with a non-zero average."""
    mask = is_cycling(rides["sport"]) & rides["has_power"] & (rides["avg_power_w"].fillna(0) > 0)
    return rides[mask]


def per_rider(rides: pd.DataFrame, cfg: AuditConfig) -> pd.DataFrame:
    """One row per rider: power-history size and weight quality."""
    pr = power_rides(rides)
    lo, hi = cfg.weight_range_kg
    g = pr.groupby("athlete_id")
    out = pd.DataFrame(
        {
            "n_power_rides": g.size(),
            "first": g["start_utc"].min(),
            "last": g["start_utc"].max(),
            "hours": g["duration_s"].sum() / 3600.0,
            "weight_median": g["weight_kg"].median(),
            "weight_n_unique": g["weight_kg"].nunique(),
            "weight_share_missing": g["weight_kg"].apply(lambda s: s.isna().mean()),
            "weight_share_plausible": g["weight_kg"].apply(lambda s: s.between(lo, hi).mean()),
            "weight_share_suspect_default": g["weight_kg"].apply(
                lambda s: s.isin(cfg.suspect_default_weights).mean()
            ),
            "mmp_20m_best": g["mmp_20m_w"].max(),
            "mmp_5s_best": g["mmp_5s_w"].max(),
        }
    )
    out["span_days"] = (out["last"] - out["first"]).dt.total_seconds() / 86400.0
    weeks = (out["span_days"] / 7.0).clip(lower=1.0)
    out["power_rides_per_week"] = out["n_power_rides"] / weeks
    out["usable"] = (out["n_power_rides"] >= cfg.min_power_rides) & (
        out["span_days"] >= cfg.min_span_days
    )
    out["weight_usable"] = (out["weight_share_plausible"] >= 0.9) & (
        out["weight_share_suspect_default"] < 0.5
    )
    return out


def implausible_values(rides: pd.DataFrame, cfg: AuditConfig) -> pd.DataFrame:
    pr = power_rides(rides)
    rows = []
    for col, limit in cfg.max_plausible_w.items():
        if col in pr:
            n = int((pr[col] > limit).sum())
            rows.append({"metric": col, "limit_w": limit, "n_above": n,
                         "share_above": n / max(len(pr), 1)})  # fmt: skip
    return pd.DataFrame(rows)


def run_audit(
    athletes: pd.DataFrame, rides: pd.DataFrame, cfg: AuditConfig | None = None
) -> dict[str, Any]:
    cfg = cfg or AuditConfig()
    cyc = rides[is_cycling(rides["sport"])]
    pr = power_rides(rides)
    riders = per_rider(rides, cfg)
    usable = riders[riders["usable"]]
    weights = pr["weight_kg"]

    overview = {
        "athletes": len(athletes),
        "activities": len(rides),
        "cycling_activities": len(cyc),
        "cycling_with_power": len(pr),
        "riders_with_any_power": int(riders.shape[0]),
        "first_activity": str(rides["start_utc"].min()) if len(rides) else "",
        "last_activity": str(rides["start_utc"].max()) if len(rides) else "",
        "share_female": float((athletes["gender"] == "F").mean()) if len(athletes) else np.nan,
        "share_csv_matched": float(rides["csv_name"].notna().mean()) if len(rides) else np.nan,
    }
    weight = {
        "range_kg": cfg.weight_range_kg,
        "power_rides_weight_missing": float(weights.isna().mean()),
        "power_rides_weight_plausible": float(weights.between(*cfg.weight_range_kg).mean()),
        "power_rides_weight_suspect_default": float(
            weights.isin(cfg.suspect_default_weights).mean()
        ),
        "most_common_weights": weights.round(1).value_counts().head(10).to_dict(),
        "usable_riders_with_usable_weight": int(usable["weight_usable"].sum()),
    }
    thresholds = {
        f">= {n} power rides": int((riders["n_power_rides"] >= n).sum())
        for n in (20, 50, 100, 200, 500)
    }
    gate = {
        "usable_riders": len(usable),
        "required": cfg.gate_min_riders,
        "passed": len(usable) >= cfg.gate_min_riders,
        "definition": f">= {cfg.min_power_rides} power rides over >= {cfg.min_span_days} days",
    }
    return {
        "overview": overview,
        "weight": weight,
        "riders_by_threshold": thresholds,
        "implausible": implausible_values(rides, cfg),
        "per_rider": riders,
        "sports": rides["sport"].fillna("").value_counts().head(15),
        "gate": gate,
    }


def report_markdown(res: dict[str, Any]) -> str:
    """Short plain-text report to paste back to Claude."""
    o, w, g = res["overview"], res["weight"], res["gate"]
    usable = res["per_rider"][res["per_rider"]["usable"]]
    lines = [
        "# GoldenCheetah OpenData audit",
        "",
        "## Overview",
        *[f"- {k}: {v}" for k, v in o.items()],
        "",
        "## Body weight (power rides)",
        f"- missing: {w['power_rides_weight_missing']:.1%}",
        f"- plausible ({w['range_kg'][0]:g}-{w['range_kg'][1]:g} kg): "
        f"{w['power_rides_weight_plausible']:.1%}",
        f"- equal to a suspected default: {w['power_rides_weight_suspect_default']:.1%}",
        f"- most common values: {w['most_common_weights']}",
        f"- usable riders whose weight looks real: {w['usable_riders_with_usable_weight']}",
        "",
        "## Riders by number of power rides",
        *[f"- {k}: {v}" for k, v in res["riders_by_threshold"].items()],
        "",
        "## Usable riders (" + g["definition"] + ")",
    ]
    if len(usable):
        desc = usable[["n_power_rides", "span_days", "hours", "power_rides_per_week",
                       "mmp_20m_best", "mmp_5s_best"]].describe().round(1)  # fmt: skip
        lines += ["```", desc.to_string(), "```"]
    lines += [
        "",
        "## Implausible values",
        "```",
        res["implausible"].to_string(index=False) if len(res["implausible"]) else "none checked",
        "```",
        "",
        "## Sports",
        "```",
        res["sports"].to_string(),
        "```",
        "",
        f"## Gate 1: {'PASSED' if g['passed'] else 'NOT PASSED'} "
        f"({g['usable_riders']} usable riders, {g['required']} required)",
    ]
    return "\n".join(lines)
