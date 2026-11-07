"""Experiment tracking with MLflow (SQLite file, no server needed).

Usage:
    from cyclingml.tracking import run
    with run("profiling", cfg) as r:
        r.log_metrics({"silhouette": 0.41})

Everything lives in paths.mlruns_dir(): `mlflow.db` (runs, params, metrics) and
`artifacts/` (files). On Colab point CYCLINGML_MLRUNS at Drive so results survive
disconnects. View them with:  mlflow ui --backend-store-uri sqlite:///<that path>/mlflow.db
"""

from __future__ import annotations

import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from cyclingml import paths
from cyclingml.config import config_hash, flatten


def _git_commit() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=paths.REPO_ROOT,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def tracking_uri() -> str:
    """SQLite database inside the MLflow folder (MLflow 3 no longer accepts plain folders)."""
    return f"sqlite:///{(paths.mlruns_dir().resolve() / 'mlflow.db').as_posix()}"


@contextmanager
def run(
    experiment: str, cfg: dict[str, Any] | None = None, run_name: str | None = None
) -> Iterator:
    """Start an MLflow run that records the config, its hash and the git commit."""
    import mlflow  # optional dependency: pip install -e ".[track]"

    store = paths.mlruns_dir().resolve()
    store.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(tracking_uri())
    if mlflow.get_experiment_by_name(experiment) is None:
        mlflow.create_experiment(experiment, artifact_location=(store / "artifacts").as_uri())
    mlflow.set_experiment(experiment)
    cfg = cfg or {}
    with mlflow.start_run(run_name=run_name) as active:
        mlflow.set_tags({"git_commit": _git_commit(), "config_hash": config_hash(cfg)})
        params = {k: str(v)[:500] for k, v in flatten(cfg).items()}
        if params:
            mlflow.log_params(params)
        yield _Run(mlflow, active)


class _Run:
    def __init__(self, mlflow_module: Any, active: Any) -> None:
        self._mlflow = mlflow_module
        self.id = active.info.run_id

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        self._mlflow.log_metrics({k: float(v) for k, v in metrics.items()}, step=step)

    def log_artifact(self, path: str) -> None:
        self._mlflow.log_artifact(path)
