import numpy as np
import pandas as pd
import pytest

from cyclingml.clean.ride import (
    CLEAN_COLUMNS,
    CleanConfig,
    clean_ride,
    fill_short_gaps,
    nan_runs,
    resample_1hz,
    spike_mask,
)
from cyclingml.config import load_config
from cyclingml.paths import REPO_ROOT


def ride(n=1200, power=200.0, step=1.0, **cols):
    """Synthetic steady ride: n samples every `step` seconds."""
    t = np.arange(n) * step
    df = pd.DataFrame({
        "time_s": t, "power_w": np.full(n, power), "hr_bpm": 140.0, "cadence_rpm": 90.0,
        "speed_mps": 8.0, "altitude_m": 300.0, "distance_m": 8.0 * t,
    })  # fmt: skip
    for k, v in cols.items():
        df[k] = v
    return df


def test_nan_runs_and_fill_short_gaps():
    x = np.array([1, np.nan, np.nan, 4, np.nan, np.nan, np.nan, np.nan, 9, np.nan])
    assert nan_runs(x) == [(1, 3), (4, 8), (9, 10)]
    y = fill_short_gaps(x, max_len=3)
    np.testing.assert_allclose(y[:4], [1, 2, 3, 4])  # short, interpolated
    assert np.isnan(y[4:8]).all()  # 4 long: not invented
    assert np.isnan(y[9])  # trailing: no right neighbour


def test_spike_mask_removes_single_spike_keeps_sprint():
    p = np.full(100, 250.0)
    p[20] = 1800  # 1-sample spike
    p[50:60] = 1100  # 10 s sprint
    m = spike_mask(p, window=7, delta_w=700)
    assert m[20] and m.sum() == 1


def test_resample_1hz_averages_and_marks_empty_seconds():
    df = ride(n=4)
    df["time_s"] = [0.0, 0.4, 1.0, 4.0]  # two samples in second 0, nothing in 2-3
    df["power_w"] = [100.0, 200.0, 300.0, 400.0]
    out, empty = resample_1hz(df)
    assert out["time_s"].tolist() == [0, 1, 2, 3, 4]
    assert out["power_w"].iloc[0] == 150
    assert empty.tolist() == [False, False, True, True, False]


def test_clean_steady_ride_is_unchanged():
    res = clean_ride(ride())
    assert res.kept and list(res.df.columns) == CLEAN_COLUMNS
    assert len(res.df) == 1200 and (res.df["power_w"] == 200).all()
    assert not res.df["is_gap"].any()
    assert res.report["power_spikes"] == 0


def test_clean_5s_recording_becomes_1hz():
    res = clean_ride(ride(n=240, step=5.0))  # old GoldenCheetah-style 5 s data
    assert res.kept and len(res.df) == 1196
    assert res.df["power_w"].notna().all() and not res.df["is_gap"].any()
    np.testing.assert_allclose(res.df["distance_m"], 8.0 * np.arange(1196))


def test_clean_long_stop_is_zero_power_gap():
    df = ride()
    df = df[(df["time_s"] < 300) | (df["time_s"] >= 600)]  # 5-minute café stop, no samples
    res = clean_ride(df)
    gap = res.df["is_gap"].to_numpy()
    assert gap.sum() == 300 and gap[300:600].all()
    assert (res.df.loc[gap, "power_w"] == 0).all()
    assert (res.df.loc[gap, "speed_mps"] == 0).all()
    assert (res.df.loc[gap, "distance_m"] == 8.0 * 299).all()  # stays where the rider stopped


def test_clean_short_gap_interpolated_and_sensor_dropout_stays_nan():
    df = ride()
    df = df[(df["time_s"] < 100) | (df["time_s"] >= 104)]  # 4 s missing: fill
    df.loc[df["time_s"].between(500, 559), "power_w"] = np.nan  # 60 s power-meter dropout
    res = clean_ride(df)
    assert not res.df["is_gap"].any()
    assert (res.df["power_w"].iloc[100:104] == 200).all()
    assert res.df["power_w"].iloc[500:560].isna().all()  # not invented, not zero
    assert res.report["power_nan_seconds"] == 60


def test_clean_keeps_coasting_zeros_and_removes_spikes():
    df = ride()
    df.loc[200:260, "power_w"] = 0.0  # descent, coasting
    df.loc[700, "power_w"] = 2000.0  # spike
    df.loc[800, "power_w"] = 4000.0  # out of range
    df.loc[900, "power_w"] = -5.0
    res = clean_ride(df)
    assert (res.df["power_w"].iloc[200:261] == 0).all()
    assert res.df["power_w"].iloc[[700, 800, 900]].tolist() == [200, 200, 200]
    assert res.report["power_spikes"] == 1 and res.report["power_out_of_range"] == 2


def test_clean_invalid_hr_cadence_speed():
    df = ride()
    df.loc[10:20, "hr_bpm"] = 0.0  # strap lost contact
    df.loc[30, "cadence_rpm"] = 255.0
    df.loc[40, "speed_mps"] = 50.0  # one GPS jump: removed, ride kept
    res = clean_ride(df)
    assert res.kept
    assert res.df["hr_bpm"].iloc[10:21].isna().all()  # 11 s > 5 s: stays NaN
    assert res.df["cadence_rpm"].iloc[30] == 90
    assert res.df["speed_mps"].iloc[40] == 8.0


@pytest.mark.parametrize(
    "df, reason",
    [(ride(n=13 * 3600 // 10, step=10.0), "longer than 12 h"),
     (ride(n=120), "shorter than 300 s"),
     (ride(speed_mps=30.0).assign(speed_mps=lambda d: np.where(d.index < 60, 40.0, 8.0)),
      "faster than"),
     (ride(n=1).iloc[:1], "no samples")],
)  # fmt: skip
def test_clean_rejects(df, reason):
    res = clean_ride(df)
    assert not res.kept and reason in res.reason


def test_config_file_matches_dataclass():
    cfg = CleanConfig.from_dict(load_config(REPO_ROOT / "configs" / "clean_rides.yaml")["clean"])
    assert cfg.max_duration_h == 12 and cfg.max_fill_gap_s == 5
    with pytest.raises(ValueError, match="unknown"):
        CleanConfig.from_dict({"max_duraton_h": 1})


def test_clean_keeps_attrs():
    df = ride()
    df.attrs["start_utc"] = "x"
    assert clean_ride(df).df.attrs["start_utc"] == "x"
