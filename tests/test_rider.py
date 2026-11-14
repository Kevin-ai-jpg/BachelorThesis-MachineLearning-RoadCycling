from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest
from fitwriter import write_fit

from cyclingml import rider
from cyclingml.config import load_config

CFG = load_config(rider.DEFAULT_CONFIG)


def fit_ride(start, minutes=20, power=200.0, sprint=None):
    recs = []
    for i in range(minutes * 60):
        w = power if power is not None else None
        if sprint and sprint[0] <= i < sprint[0] + sprint[1]:
            w = sprint[2]
        recs.append(dict(time=start + timedelta(seconds=i), power=w, heart_rate=140,
                         cadence=90, distance=8.0 * i, enhanced_speed=8.0))  # fmt: skip
    return write_fit(recs)


def make_export(root):
    d0 = datetime(2026, 3, 1, 9, tzinfo=timezone.utc)
    act = root / "activities"
    act.mkdir(parents=True)
    (act / "1.fit").write_bytes(fit_ride(d0, sprint=(300, 10, 1000.0)))
    (act / "2.fit").write_bytes(fit_ride(d0 + timedelta(days=7), power=250.0))
    (act / "3.fit").write_bytes(fit_ride(d0 + timedelta(days=8), power=None))  # no power meter
    (act / "4.fit").write_bytes(fit_ride(d0 + timedelta(days=120), power=180.0))
    (root / "garmin").mkdir()
    (root / "garmin" / "dup.fit").write_bytes(fit_ride(d0 + timedelta(seconds=30), power=None))
    (act / "broken.fit").write_bytes(b"garbage")


def test_process_amateur_export(tmp_path):
    make_export(tmp_path / "export")
    acts = rider.process_rider(rider.iter_amateur(tmp_path / "export"), "r1", CFG,
                               rides_dir=tmp_path / "rides")  # fmt: skip
    status = acts["status"].value_counts().to_dict()
    assert status == {"kept": 4, "duplicate": 1, "unreadable": 1}
    # the power-less copy of ride 1 is the duplicate, not ride 1 itself
    assert acts.loc[acts["status"] == "duplicate", "source"].str.contains("dup").all()
    kept = acts[acts["status"] == "kept"].set_index("source")
    r1 = kept.loc["activities/1.fit"]
    assert r1["mmp_5s"] == 1000 and r1["mmp_10s"] == 1000
    assert r1["mmp_60s"] == (10 * 1000 + 50 * 200) / 60
    assert np.isnan(r1["mmp_1800s"])  # 20-minute ride
    assert not kept["has_power"].iloc[2]  # ride 3, no power
    # rolling 90-day best: ride 2 still sees ride 1's sprint, ride 4 (120 days later) does not
    best = acts[acts["has_power"].eq(True)]["best90_mmp_5s"].tolist()
    assert best == [1000, 1000, 180]
    assert len(list((tmp_path / "rides").glob("*.parquet"))) == 4

    md = rider.summary_markdown(acts, "r1", weight_kg=80)
    assert "| 5 s | 1000 | 180 | 12.50 | 2.25 |" in md
    assert "unreadable: 1" in md


def test_main_writes_outputs(tmp_path, capsys):
    make_export(tmp_path / "export")
    rider.main([str(tmp_path / "export"), "--rider-id", "r1", "--out", str(tmp_path / "out")])
    assert (tmp_path / "out" / "summary.md").exists()
    acts = pd.read_parquet(tmp_path / "out" / "activities.parquet")
    assert len(acts) == 6
    assert "Rider r1" in capsys.readouterr().out


def test_goldencheetah_zip(athlete_zip):
    acts = rider.process_rider(rider.iter_goldencheetah(athlete_zip), "gc1", CFG)
    # the fixture's rides are 4 s long: read, then rejected as too short
    assert len(acts) == 3
    assert (acts["status"] == "rejected").all()
    assert acts["reason"].str.contains("shorter").all()
    assert acts["gc_mmp_20m_w"].eq(250).all()  # GoldenCheetah's own value kept for comparison


def test_gc_check_ratio():
    from cyclingml.eval.gc_check import compare

    acts = pd.DataFrame({"status": ["kept", "kept", "rejected"], "has_power": [True, True, None],
                         **{f"mmp_{s}s": [100.0, 99.0, 1.0] for s in (5, 60, 300, 1200, 3600)},
                         **{f"gc_mmp_{n}_w": [100.0, 110.0, 5.0]
                            for n in ("5s", "1m", "5m", "20m", "60m")}})  # fmt: skip
    out = compare(acts, "r").set_index("duration")
    assert out.loc["5s", "rides"] == 2
    assert out.loc["5s", "median_ratio"] == pytest.approx((1 + 0.9) / 2)
    assert out.loc["20m", "within_2pct"] == 0.5
