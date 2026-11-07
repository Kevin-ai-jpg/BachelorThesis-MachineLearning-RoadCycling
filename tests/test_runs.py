import pytest

pytest.importorskip("mlflow")

from cyclingml.runs import experiments, runs_table  # noqa: E402
from cyclingml.tracking import run  # noqa: E402


def test_runs_table_sorts_by_metric(tmp_path, monkeypatch):
    monkeypatch.setenv("CYCLINGML_MLRUNS", str(tmp_path / "mlruns"))
    for k, score in [(3, 0.2), (5, 0.6), (8, 0.4)]:
        with run("profiling-test", {"model": {"k": k}}, run_name=f"k{k}") as r:
            r.log_metrics({"silhouette": score})
    assert "profiling-test" in experiments()["experiment"].tolist()
    table = runs_table("profiling-test", metric="silhouette")
    assert table["run"].tolist() == ["k5", "k8", "k3"]
    assert table["model.k"].tolist() == ["5", "8", "3"]
    assert runs_table("profiling-test", metric="silhouette", ascending=True)["run"].iloc[0] == "k3"
