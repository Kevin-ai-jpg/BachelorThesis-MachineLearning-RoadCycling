"""Experiment D3: recover unlabelled GoldenCheetah activities as rides.

    python -m cyclingml.eval.sport_rule configs/data_sport_rule.yaml [--rides <cached parquet>]

1. Pool: labelled activities (bike or certainly-not-bike label) that have power, since only rides
   with power are useful for training. Truth = bike label.
2. Split by rider (50/50, fixed seed). Choose the rule's thresholds on TRAIN riders (highest recall
   with precision >= target), report precision / recall / false-positive rate on TEST riders,
   with a 95% bootstrap interval over test riders. Baselines: "has power" alone, and a shallow
   decision tree trained on the same train riders.
3. Apply the chosen rule to every activity with an unknown label and recount Gate 1 usable riders
   (>= 100 power rides over >= 180 days), separately for label aliases (e.g. "Rennrad") and for the
   rule.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cyclingml import paths
from cyclingml.clean import sport as sp
from cyclingml.config import load_config
from cyclingml.io.gc_audit import AuditConfig, per_rider
from cyclingml.io.goldencheetah import load_index
from cyclingml.tracking import run


def split_riders(ids: pd.Series, test_share: float, seed: int) -> pd.Series:
    """True for activities of held-out (test) riders."""
    riders = np.sort(ids.unique())
    rng = np.random.default_rng(seed)
    test = set(rng.choice(riders, size=int(round(len(riders) * test_share)), replace=False))
    return ids.isin(test)


def bootstrap_ci(pred, truth, rider, n=200, seed=0) -> dict[str, float]:
    """95% interval of precision and recall, resampling test riders with replacement."""
    df = pd.DataFrame({"tp": pred & truth, "fp": pred & ~truth, "fn": ~pred & truth, "r": rider})
    per = df.groupby("r")[["tp", "fp", "fn"]].sum().to_numpy(float)
    rng = np.random.default_rng(seed)
    prec, rec = [], []
    for _ in range(n):
        s = per[rng.integers(0, len(per), len(per))].sum(axis=0)
        prec.append(s[0] / max(s[0] + s[1], 1))
        rec.append(s[0] / max(s[0] + s[2], 1))
    return {"precision_lo": float(np.percentile(prec, 2.5)),
            "precision_hi": float(np.percentile(prec, 97.5)),
            "recall_lo": float(np.percentile(rec, 2.5)),
            "recall_hi": float(np.percentile(rec, 97.5))}  # fmt: skip


def choose_rule(
    train: pd.DataFrame, truth: pd.Series, cfg: dict[str, Any]
) -> tuple[sp.Rule, pd.DataFrame]:
    rc = cfg["rule"]
    rows = []
    for vmin, indoor in itertools.product(rc["grid"]["speed_min_mps"], rc["grid"]["allow_indoor"]):
        rule = sp.Rule(speed_min_mps=vmin, allow_indoor=indoor, speed_max_mps=rc["speed_max_mps"],
                       indoor_speed_mps=rc["indoor_speed_mps"],
                       cadence_range_rpm=tuple(rc["cadence_range_rpm"]),
                       min_avg_power_w=rc["min_avg_power_w"])  # fmt: skip
        m = sp.precision_recall(sp.rule_is_ride(train, rule), truth)
        rows.append({"speed_min_mps": vmin, "allow_indoor": indoor, **m, "rule": rule})
    grid = pd.DataFrame(rows)
    ok = grid[grid["precision"] >= rc["precision_target"]]
    best = (ok if len(ok) else grid).sort_values(["recall", "precision"], ascending=False).iloc[0]
    return best["rule"], grid.drop(columns="rule")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("config", type=Path)
    p.add_argument("--rides", type=Path, default=None, help="cached rides parquet (faster)")
    p.add_argument("--out", type=Path, default=None, help="folder for tables (default: scratch)")
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    out = args.out or paths.processed_dir() / "sport_rule"
    out.mkdir(parents=True, exist_ok=True)

    if args.rides:
        rides = pd.read_parquet(args.rides)
    else:
        _, rides = load_index(paths.data_root() / cfg["index_dir"])
    lab = cfg["labels"]
    rides["label_class"] = sp.label_class(rides["sport"], lab["bike"], lab["nonbike"])
    with_power = sp.has_power_baseline(rides, cfg["rule"]["min_avg_power_w"])

    pool = rides[with_power & rides["label_class"].isin([sp.BIKE, sp.NONBIKE])]
    truth = pool["label_class"] == sp.BIKE
    is_test = split_riders(pool["athlete_id"], cfg["test_share"], cfg["seed"])
    train, test = pool[~is_test], pool[is_test]
    print(
        f"pool: {len(pool)} labelled power activities, {truth.mean():.1%} bike; "
        f"train riders {train.athlete_id.nunique()}, test riders {test.athlete_id.nunique()}"
    )

    rule, grid = choose_rule(train, truth[~is_test], cfg)
    print("grid on TRAIN riders:\n", grid.round(4).to_string(index=False))

    from sklearn.tree import DecisionTreeClassifier, export_text

    tree = DecisionTreeClassifier(max_depth=cfg["tree"]["max_depth"],
                                  min_samples_leaf=cfg["tree"]["min_samples_leaf"],
                                  random_state=cfg["seed"])  # fmt: skip
    Xtr, Xte = sp.tree_features(train), sp.tree_features(test)
    tree.fit(Xtr, truth[~is_test])

    preds = {
        "baseline_has_power": pd.Series(True, index=test.index),
        "rule": sp.rule_is_ride(test, rule),
        "tree": pd.Series(tree.predict(Xte).astype(bool), index=test.index),
    }
    results = {}
    for name, pred in preds.items():
        m = sp.precision_recall(pred, truth[is_test])
        m.update(bootstrap_ci(pred.to_numpy(), truth[is_test].to_numpy(),
                              test["athlete_id"].to_numpy(), seed=cfg["seed"]))  # fmt: skip
        results[name] = m
        params = {**cfg, "method": name,
                  "chosen_rule": {"speed_min_mps": rule.speed_min_mps,
                                  "allow_indoor": rule.allow_indoor}}  # fmt: skip
        with run(cfg["experiment"], params, run_name=f"d3-{name}") as r:
            r.log_metrics({k: v for k, v in m.items() if np.isfinite(v)})
    res = pd.DataFrame(results).T
    print("TEST riders:\n", res.round(4).to_string())

    # errors of the rule on test: which labels does it accept wrongly?
    fp = test[preds["rule"] & ~truth[is_test]]
    fp_labels = sp.normalise_label(fp["sport"]).value_counts().head(10)
    print("rule false positives by label (test):\n", fp_labels.to_string())

    # --- apply: aliases + rule on unknown labels
    audit = AuditConfig()
    before = per_rider(rides, audit)
    alias = rides["label_class"] == sp.BIKE
    unknown = rides["label_class"] == sp.UNKNOWN
    recovered = unknown & sp.rule_is_ride(rides, rule)
    empty = sp.normalise_label(rides["sport"]) == ""

    def usable_with(mask: pd.Series) -> pd.DataFrame:
        r2 = rides[["athlete_id", "start_utc", "has_power", "avg_power_w", "duration_s",
                    "weight_kg", "mmp_20m_w", "mmp_5s_w"]].copy()  # fmt: skip
        r2["sport"] = np.where(mask, "bike", "other")
        return per_rider(r2, audit)

    after_alias = usable_with(alias)
    after_all = usable_with(alias | recovered)
    gain = {
        "power_rides_before": int(before["n_power_rides"].sum()),
        "power_rides_alias": int(after_alias["n_power_rides"].sum()),
        "power_rides_alias_rule": int(after_all["n_power_rides"].sum()),
        "unknown_with_power": int((unknown & with_power).sum()),
        "recovered_by_rule": int(recovered.sum()),
        "recovered_empty_label": int((recovered & empty).sum()),
        "usable_riders_before": int(before["usable"].sum()),
        "usable_riders_alias": int(after_alias["usable"].sum()),
        "usable_riders_alias_rule": int(after_all["usable"].sum()),
        "usable_weight_before": int((before["usable"] & before["weight_usable"]).sum()),
        "usable_weight_alias_rule": int((after_all["usable"] & after_all["weight_usable"]).sum()),
    }
    with run(cfg["experiment"], {**cfg, "method": "apply-rule"}, run_name="d3-apply") as r:
        r.log_metrics(gain)
    print(json.dumps(gain, indent=1))

    grid.to_csv(out / "grid_train.csv", index=False)
    res.to_csv(out / "test_metrics.csv")
    (out / "tree.txt").write_text(export_text(tree, feature_names=list(Xtr.columns)),
                                  encoding="utf-8")  # fmt: skip
    (out / "gain.json").write_text(json.dumps(gain, indent=1), encoding="utf-8")
    print(f"tables in {out}")


if __name__ == "__main__":
    main()
