import pytest

mlflow = pytest.importorskip("mlflow")

from cyclingml.tracking import run, tracking_uri  # noqa: E402


def test_dummy_run_is_logged(tmp_path, monkeypatch):
    monkeypatch.setenv("CYCLINGML_MLRUNS", str(tmp_path / "mlruns"))
    cfg = {"model": {"name": "dummy", "k": 3}}
    with run("smoke-test", cfg, run_name="dummy") as r:
        r.log_metrics({"score": 0.5})
        run_id = r.id

    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri())
    logged = client.get_run(run_id)
    assert logged.data.metrics["score"] == 0.5
    assert logged.data.params["model.k"] == "3"
    assert "config_hash" in logged.data.tags
