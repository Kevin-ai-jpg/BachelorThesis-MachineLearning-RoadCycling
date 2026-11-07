"""Shared fixtures: a synthetic GoldenCheetah OpenData athlete zip with known contents."""

from __future__ import annotations

import json
import zipfile
from datetime import datetime, timedelta

import pytest

ATHLETE_ID = "033874ce-e20d-44ba-9cc9-125030b6662f"


def _ride(date_utc: datetime, sport="Bike", flags="TDSPHC-AGL-----", weight="80.0", power=200.0):
    return {
        "date": date_utc.strftime("%Y/%m/%d %H:%M:%S UTC"),
        "data": flags,
        "sport": sport,
        "METRICS": {
            "workout_time": "3600.00000",
            "time_riding": "3500.00000",
            "average_power": [f"{power:.5f}", "3600.00000"],
            "coggan_tss": "55.00000",
            "athlete_weight": weight,
            "average_temp": "-255.00000",
            "5s_critical_power": "600.00000",
            "20m_critical_power": "250.00000",
            "power_dist": [1, 2, 3],
        },
    }


def make_athlete_zip(path, n_power_rides=3, start=datetime(2020, 1, 1, 10, 0, 7), tz_hours=2):
    """Zip with n power rides one day apart (each CSV named in local time, UTC+tz_hours),
    one run without a CSV and one ride with the -255 sentinel weight."""
    rides, csvs = [], {}
    for i in range(n_power_rides):
        t = start + timedelta(days=i)
        rides.append(_ride(t))
        local = t + timedelta(hours=tz_hours)
        csvs[local.strftime("%Y_%m_%d_%H_%M_%S") + ".csv"] = (
            "secs,km,power,hr,cad,alt\n0,0,,100,,67\n1,0.010,200,101,85,68\n"
            "2,0.020,210,102,86,69\n4,0.040,220,103,87,70\n"
        )
    rides.append(_ride(start + timedelta(days=n_power_rides), sport="Run", flags="TDS-H----------"))
    rides.append(_ride(start + timedelta(days=n_power_rides + 1), weight="-255.00000"))
    meta = {
        "VERSION": "1.8",
        "ATHLETE": {"gender": "M", "yob": "1967", "id": "{" + ATHLETE_ID + "}"},
        "RIDES": rides,
    }
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("{" + ATHLETE_ID + "}.json", json.dumps(meta))
        for name, body in csvs.items():
            zf.writestr(name, body)
    return path


@pytest.fixture
def athlete_zip(tmp_path):
    return make_athlete_zip(tmp_path / f"{ATHLETE_ID}.zip")
