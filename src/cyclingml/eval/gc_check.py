"""Check our cleaned power curves against GoldenCheetah's own per-activity MMP metrics.

For riders processed with `python -m cyclingml.rider <zip> --goldencheetah --out <dir>/<rider>`:

    python -m cyclingml.eval.gc_check <dir>

For every power ride, ratio = our best power / GoldenCheetah's at 5 s, 1, 5, 20 and 60 min.
Ratio ~1 means our reader + cleaning + power curve reproduce GoldenCheetah; large deviations
point at either our cleaning (spikes removed) or at broken GoldenCheetah metadata.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

GC_DURATIONS = {"5s": 5, "1m": 60, "5m": 300, "20m": 1200, "60m": 3600}


def compare(acts: pd.DataFrame, rider: str) -> pd.DataFrame:
    p = acts[(acts["status"] == "kept") & acts["has_power"].eq(True)]
    rows = []
    for name, sec in GC_DURATIONS.items():
        ours, gc = p[f"mmp_{sec}s"], p[f"gc_mmp_{name}_w"]
        ok = ours.notna() & gc.notna() & (gc > 0)
        r = ours[ok] / gc[ok]
        rows.append({"rider": rider, "duration": name, "rides": int(ok.sum()),
                     "median_ratio": r.median(), "within_2pct": (abs(r - 1) <= 0.02).mean(),
                     "best_ours_w": ours.max(), "best_gc_w": gc.max()})  # fmt: skip
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    root = Path((argv or sys.argv[1:])[0])
    frames = [compare(pd.read_parquet(f), f.parent.name)
              for f in sorted(root.glob("*/activities.parquet"))]  # fmt: skip
    df = pd.concat(frames, ignore_index=True)
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
