import gzip
import io
import zipfile
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from fitwriter import write_fit

from cyclingml.io import activity as act
from cyclingml.io.goldencheetah import RIDE_SAMPLE_COLUMNS

T0 = datetime(2026, 5, 1, 8, 0, 0, tzinfo=timezone.utc)


def fit_records(n=5):
    recs = [dict(time=T0 + timedelta(seconds=i), power=200 + i, heart_rate=120 + i, cadence=90,
                 distance=8.0 * i, enhanced_speed=8.0, enhanced_altitude=300.4,
                 lat=46.77, lon=23.6 + i * 1e-4) for i in range(n)]  # fmt: skip
    recs[2]["power"] = None  # dropout -> NaN
    return recs


GPX = """<?xml version="1.0" encoding="UTF-8"?>
<gpx version="1.1" creator="test" xmlns="http://www.topografix.com/GPX/1/1"
 xmlns:gpxtpx="http://www.garmin.com/xmlschemas/TrackPointExtension/v1">
 <trk><trkseg>
  <trkpt lat="46.0" lon="23.0"><ele>300</ele><time>2026-05-01T08:00:00Z</time>
   <extensions><power>250</power><gpxtpx:TrackPointExtension><gpxtpx:hr>130</gpxtpx:hr>
   <gpxtpx:cad>88</gpxtpx:cad></gpxtpx:TrackPointExtension></extensions></trkpt>
  <trkpt lat="46.0001" lon="23.0"><ele>301</ele><time>2026-05-01T08:00:02Z</time>
   <extensions><power>260</power><gpxtpx:TrackPointExtension><gpxtpx:hr>131</gpxtpx:hr>
   </gpxtpx:TrackPointExtension></extensions></trkpt>
 </trkseg></trk></gpx>"""

TCX = """
<?xml version="1.0" encoding="UTF-8"?>
<TrainingCenterDatabase xmlns="http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2"
 xmlns:ns3="http://www.garmin.com/xmlschemas/ActivityExtension/v2">
 <Activities><Activity Sport="Biking"><Lap StartTime="2026-05-01T08:00:00Z"><Track>
  <Trackpoint><Time>2026-05-01T08:00:00Z</Time><AltitudeMeters>300</AltitudeMeters>
   <DistanceMeters>0</DistanceMeters><HeartRateBpm><Value>120</Value></HeartRateBpm>
   <Cadence>90</Cadence><Extensions><ns3:TPX><ns3:Speed>8.5</ns3:Speed>
   <ns3:Watts>210</ns3:Watts></ns3:TPX></Extensions></Trackpoint>
  <Trackpoint><Time>2026-05-01T08:00:01Z</Time><AltitudeMeters>300.5</AltitudeMeters>
   <DistanceMeters>8.5</DistanceMeters><HeartRateBpm><Value>121</Value></HeartRateBpm>
   <Cadence>91</Cadence><Extensions><ns3:TPX><ns3:Speed>8.6</ns3:Speed>
   <ns3:Watts>220</ns3:Watts></ns3:TPX></Extensions></Trackpoint>
 </Track></Lap></Activity></Activities></TrainingCenterDatabase>"""


def test_activity_format():
    assert act.activity_format("a/Ride.FIT") == "fit"
    assert act.activity_format("x.tcx.gz") == "tcx"
    assert act.activity_format("x.gpx") == "gpx"
    assert act.activity_format("x.csv") is None


def test_haversine_one_degree_latitude():
    d = act.haversine_m(np.array([0.0, 1.0]), np.array([0.0, 0.0]))
    assert d[0] == 0
    assert d[1] == pytest.approx(111_195, rel=1e-3)  # 2*pi*6371 km / 360


def test_read_fit():
    df = act.read_activity_bytes(write_fit(fit_records()), "ride.fit")
    assert list(df.columns) == RIDE_SAMPLE_COLUMNS
    assert df["time_s"].tolist() == [0, 1, 2, 3, 4]
    assert df["power_w"].tolist()[:2] == [200, 201] and np.isnan(df.loc[2, "power_w"])
    assert df["hr_bpm"].iloc[-1] == 124
    assert df["speed_mps"].eq(8.0).all()
    assert df["altitude_m"].iloc[0] == pytest.approx(300.4)
    assert df["distance_m"].iloc[-1] == pytest.approx(32.0)
    assert df.attrs["start_utc"] == T0 and df.attrs["format"] == "fit"
    assert "lat" not in df.columns  # no coordinates leave the reader


def test_read_fit_gzipped_and_without_distance():
    recs = fit_records()
    for r in recs:
        r["distance"] = None
        r["enhanced_speed"] = None
    df = act.read_activity_bytes(gzip.compress(write_fit(recs)), "ride.fit.gz")
    # 1e-4 degrees of longitude at 46.77 N ~ 7.6 m per second, derived from GPS
    assert df["distance_m"].iloc[-1] == pytest.approx(4 * 7.62, rel=0.01)
    assert df["speed_mps"].iloc[1:].between(7.5, 7.7).all()


def test_read_gpx():
    df = act.read_activity_bytes(GPX.encode(), "ride.gpx")
    assert df["time_s"].tolist() == [0, 2]
    assert df["power_w"].tolist() == [250, 260]
    assert df["hr_bpm"].tolist() == [130, 131]
    assert df.loc[0, "cadence_rpm"] == 88 and np.isnan(df.loc[1, "cadence_rpm"])
    assert df.loc[1, "distance_m"] == pytest.approx(11.12, rel=1e-2)  # 1e-4 deg latitude
    assert df.loc[1, "speed_mps"] == pytest.approx(5.56, rel=1e-2)


def test_read_tcx_with_leading_whitespace():
    df = act.read_activity_bytes(TCX.encode(), "ride.tcx")
    assert df["power_w"].tolist() == [210, 220]
    assert df["speed_mps"].tolist() == [8.5, 8.6]
    assert df["distance_m"].tolist() == [0, 8.5]
    assert df["hr_bpm"].tolist() == [120, 121]
    assert df.attrs["start_utc"] == T0


def test_corrupt_and_unsupported_files():
    with pytest.raises(act.UnsupportedActivity):
        act.read_activity_bytes(b"not a fit file", "x.fit")
    with pytest.raises(act.UnsupportedActivity):
        act.read_activity_bytes(b"a,b\n1,2", "x.csv")


def test_iter_activity_files_reads_nested_zips(tmp_path):
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w") as z:
        z.writestr("123.fit", write_fit(fit_records()))
        z.writestr("notes.txt", "ignore me")
    with zipfile.ZipFile(tmp_path / "export.zip", "w") as z:
        z.writestr("DI_CONNECT/UploadedFiles_0-_Part1.zip", inner.getvalue())
    (tmp_path / "activities").mkdir()
    (tmp_path / "activities" / "1.gpx").write_text(GPX)
    (tmp_path / "activities" / "2.tcx.gz").write_bytes(gzip.compress(TCX.encode()))
    names = [n for n, _ in act.iter_activity_files(tmp_path)]
    assert len(names) == 3
    assert any(n.endswith("UploadedFiles_0-_Part1.zip/123.fit") for n in names)
    frames = [act.read_activity_bytes(b, n) for n, b in act.iter_activity_files(tmp_path)]
    assert {f.attrs["format"] for f in frames} == {"fit", "gpx", "tcx"}
