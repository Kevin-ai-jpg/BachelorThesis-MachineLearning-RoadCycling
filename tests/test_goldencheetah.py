import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest
from conftest import ATHLETE_ID, make_athlete_zip

from cyclingml.io import goldencheetah as gc
from cyclingml.io.gc_audit import AuditConfig, report_markdown, run_audit


@pytest.mark.parametrize(
    "raw, expected",
    [("12.5", 12.5), (["7.00000", "3600.0"], 7.0), ("-255.00000", math.nan), ("nan", math.nan),
     (None, math.nan), ([1, 2, 3], math.nan), ("abc", math.nan)],
)  # fmt: skip
def test_metric_value(raw, expected):
    got = gc.metric_value(raw)
    assert (math.isnan(got) and math.isnan(expected)) or got == expected


def test_parse_date_is_utc():
    d = gc.parse_date("2017/07/18 11:01:20 UTC")
    assert d == datetime(2017, 7, 18, 11, 1, 20, tzinfo=timezone.utc)


def test_channel_flags():
    assert gc.has_channel("TDSPHC-AGL-----", "power")
    assert not gc.has_channel("TDS-HC-A-L-----", "power")
    assert gc.has_channel("TDS-HC-A-L-----", "altitude")
    assert not gc.has_channel("", "power")


def test_match_csv_handles_timezone_offset():
    ride = datetime(2020, 1, 1, 10, 0, 7, tzinfo=timezone.utc)
    names = {
        "2020_01_01_12_00_07.csv": datetime(2020, 1, 1, 12, 0, 7),  # UTC+2, right one
        "2020_01_01_12_00_08.csv": datetime(2020, 1, 1, 12, 0, 8),  # different second
        "2020_01_02_12_00_07.csv": datetime(2020, 1, 2, 12, 0, 7),  # next day
    }
    assert gc.match_csv(ride, names) == "2020_01_01_12_00_07.csv"
    assert gc.match_csv(ride, {"2020_01_03_10_00_07.csv": datetime(2020, 1, 3, 10, 0, 7)}) is None


def test_ride_table(athlete_zip):
    athlete, rides = gc.ride_table(athlete_zip)
    assert athlete["athlete_id"] == ATHLETE_ID
    assert athlete["gender"] == "M" and athlete["yob"] == 1967
    assert athlete["n_activities"] == 5 and athlete["n_csv"] == 3
    assert len(rides) == 5
    assert rides["has_power"].sum() == 4  # 3 + the sentinel-weight ride; the run has none
    assert rides["csv_name"].notna().sum() == 3
    assert rides.loc[0, "avg_power_w"] == 200.0
    assert rides.loc[0, "mmp_20m_w"] == 250.0
    assert np.isnan(rides.iloc[-1]["weight_kg"])  # -255 sentinel
    assert np.isnan(rides.loc[0, "mmp_1m_w"])  # absent metric


def test_read_ride_csv(athlete_zip):
    _, rides = gc.ride_table(athlete_zip)
    df = gc.read_ride_csv(athlete_zip, rides.loc[0, "csv_name"])
    assert list(df.columns) == gc.RIDE_SAMPLE_COLUMNS
    assert df["time_s"].tolist() == [0, 1, 2, 4]
    assert np.isnan(df.loc[0, "power_w"]) and df.loc[1, "power_w"] == 200
    assert df.loc[1, "distance_m"] == pytest.approx(10.0)
    assert df.loc[3, "speed_mps"] == pytest.approx(10.0)  # 20 m in 2 s (gap in samples)
    assert np.isnan(df.loc[0, "speed_mps"])


def test_build_index_and_audit(tmp_path):
    zips = tmp_path / "zips"
    zips.mkdir()
    make_athlete_zip(zips / "a.zip", n_power_rides=5)
    (zips / "broken.zip").write_bytes(b"not a zip")
    out = tmp_path / "index"
    summary = gc.build_index(zips, out, workers=1)
    assert summary == {"zips": 2, "athletes": 1, "errors": 1, "parts": 1}
    assert "broken.zip" in (out / "index_errors.txt").read_text()

    athletes, rides = gc.load_index(out)
    assert len(athletes) == 1 and len(rides) == 7
    assert isinstance(rides["start_utc"].dtype, pd.DatetimeTZDtype)

    cfg = AuditConfig(min_power_rides=5, min_span_days=3, gate_min_riders=1)
    res = run_audit(athletes, rides, cfg)
    assert res["overview"]["cycling_with_power"] == 6
    rider = res["per_rider"].iloc[0]
    assert rider["n_power_rides"] == 6 and rider["usable"]
    assert rider["weight_share_missing"] == pytest.approx(1 / 6)
    assert res["gate"]["passed"]
    text = report_markdown(res)
    assert "Gate 1: PASSED" in text and "most common values" in text


def test_audit_flags_default_weight(tmp_path):
    zips = tmp_path / "zips"
    zips.mkdir()
    make_athlete_zip(zips / "a.zip", n_power_rides=4)
    out = tmp_path / "index"
    gc.build_index(zips, out, workers=1)
    athletes, rides = gc.load_index(out)
    rides["weight_kg"] = 75.0
    res = run_audit(athletes, rides, AuditConfig(min_power_rides=1, min_span_days=0))
    assert res["weight"]["power_rides_weight_suspect_default"] == 1.0
    assert not res["per_rider"].iloc[0]["weight_usable"]
