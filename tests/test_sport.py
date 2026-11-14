import numpy as np
import pandas as pd
import pytest

from cyclingml.clean import sport as sp
from cyclingml.eval.sport_rule import split_riders


def acts():
    # outdoor ride, indoor ride (no distance), Stryd run, rowing erg, no power, car (too fast)
    return pd.DataFrame({
        "sport": ["Rennrad", "", " RUN ", "Rowing", "", "Workout"],
        "has_power": [True, True, True, True, False, True],
        "avg_power_w": [180.0, 200.0, 230.0, 150.0, np.nan, 150.0],
        "avg_cadence_rpm": [85.0, 90.0, 84.0, 24.0, 80.0, np.nan],
        "distance_km": [60.0, 0.0, 10.0, 0.0, 30.0, 300.0],
        "moving_time_s": [7200.0, 3600.0, 3300.0, 1800.0, 3600.0, 7200.0],
    })  # fmt: skip


def test_label_class():
    c = sp.label_class(acts()["sport"], bike=["rennrad"], nonbike=["run", "rowing"])
    assert c.tolist() == ["bike", "unknown", "nonbike", "nonbike", "unknown", "unknown"]


def test_avg_speed():
    v = sp.avg_speed_mps(acts())
    assert v[0] == pytest.approx(60000 / 7200) and v[1] == 0


def test_rule():
    got = sp.rule_is_ride(acts(), sp.Rule())
    # ride yes, indoor yes, run no (3 m/s), rowing no (cadence 24), no power no, car no
    assert got.tolist() == [True, True, False, False, False, False]
    no_indoor = sp.rule_is_ride(acts(), sp.Rule(allow_indoor=False))
    assert no_indoor.tolist() == [True, False, False, False, False, False]


def test_precision_recall():
    m = sp.precision_recall(pd.Series([1, 1, 0, 1]), pd.Series([1, 0, 1, 1]))
    assert m["precision"] == pytest.approx(2 / 3) and m["recall"] == pytest.approx(2 / 3)
    assert m["fpr"] == 1.0 and (m["tp"], m["fp"], m["fn"]) == (2, 1, 1)


def test_split_is_by_rider():
    ids = pd.Series(["a", "a", "b", "c", "c", "d"])
    t = split_riders(ids, 0.5, seed=0)
    assert t.groupby(ids).nunique().eq(1).all()  # a rider is never on both sides
    assert ids[t].nunique() == 2
    assert t.equals(split_riders(ids, 0.5, seed=0))  # reproducible


def test_tree_features_have_no_nan():
    a = acts().assign(elevation_gain_m=1.0, duration_s=1.0, has_gps=True, has_hr=True, np_w=200.0)
    assert not sp.tree_features(a).isna().any().any()
