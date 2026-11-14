import numpy as np
import pandas as pd
import pytest

from cyclingml.features import power_curve as pc


def test_known_answer():
    p = np.array([100, 100, 500, 400, 100, 100, 100, 100], float)
    got = pc.best_powers(p, [1, 2, 3, 8, 9])
    np.testing.assert_allclose(got[:4], [500, 450, 1000 / 3, 1500 / 8])
    assert np.isnan(got[4])  # longer than the ride


def test_constant_power_curve_is_flat():
    got = pc.best_powers(np.full(4000, 250.0), pc.DEFAULT_DURATIONS_S)
    assert np.allclose(got[np.isfinite(got)], 250)
    assert np.isnan(got[-1])  # 3 h > 4000 s


@pytest.mark.parametrize("seed", range(5))
def test_matches_bruteforce_on_random_rides(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(50, 700))
    p = rng.gamma(2.0, 100.0, n)
    p[rng.random(n) < 0.1] = 0  # coasting
    p[rng.random(n) < 0.02] = np.nan  # dropouts
    durs = [1, 2, 5, 10, 30, 60, 120, 300, 600]
    np.testing.assert_allclose(pc.best_powers(p, durs), pc.best_powers_bruteforce(p, durs))


def test_multiples_of_a_duration_never_beat_it():
    # best(k*d) <= best(d): a k*d window splits into k windows of d. (Plain monotonicity does
    # NOT hold: for [10, 0, 10] best(2 s) = 5 W < best(3 s) = 6.7 W.)
    rng = np.random.default_rng(0)
    durs = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    got = pc.best_powers(rng.gamma(2.0, 100.0, 11000), durs)
    assert np.all(np.diff(got) <= 1e-9)
    np.testing.assert_allclose(pc.best_powers(np.array([10.0, 0, 10]), [2, 3]), [5, 20 / 3])


def test_activity_curve_columns():
    ride = pd.DataFrame({"power_w": np.full(100, 300.0)})
    c = pc.activity_curve(ride, [1, 60])
    assert c == {"mmp_1s": 300.0, "mmp_60s": 300.0}


def curves():
    days = pd.to_datetime(["2026-01-01", "2026-02-01", "2026-05-15", "2026-05-16"], utc=True)
    return pd.DataFrame({
        "rider_id": ["a", "a", "a", "a"], "start_utc": days,
        "mmp_5s": [900.0, 800.0, 700.0, 750.0], "mmp_1200s": [250.0, 270.0, 240.0, 230.0],
    })  # fmt: skip


def test_rolling_best_including_current():
    out = pc.rolling_best(curves(), window_days=90)
    assert out["mmp_5s"].tolist() == [900, 900, 700, 750]  # Jan and Feb are > 90 days before
    assert out["mmp_1200s"].tolist() == [250, 270, 240, 240]


def test_rolling_best_past_only_has_no_leakage():
    out = pc.rolling_best(curves(), window_days=90, include_current=False)
    assert np.isnan(out["mmp_5s"].iloc[0])  # nothing before the first ride
    assert np.isnan(out["mmp_5s"].iloc[2])  # nothing in the 90 days before May 15
    assert out["mmp_5s"].tolist()[1] == 900 and out["mmp_5s"].tolist()[3] == 700
    # leakage guard: the value at each ride never depends on that ride or later ones
    c = curves()
    c.loc[3, ["mmp_5s", "mmp_1200s"]] = 5000.0
    again = pc.rolling_best(c, window_days=90, include_current=False)
    pd.testing.assert_frame_equal(again.iloc[:4][["mmp_5s"]], out.iloc[:4][["mmp_5s"]])


def test_rolling_best_separates_riders_and_keeps_order():
    c = pd.concat([curves(), curves().assign(rider_id="b", mmp_5s=100.0)], ignore_index=True)
    c = c.sample(frac=1, random_state=0)  # shuffled input
    out = pc.rolling_best(c)
    assert (out.index == c.index).all()
    assert (out.loc[out["rider_id"] == "b", "mmp_5s"] == 100).all()
    best = pc.alltime_best(c)
    assert best.loc["a", "mmp_5s"] == 900 and best.loc["b", "mmp_1200s"] == 270
