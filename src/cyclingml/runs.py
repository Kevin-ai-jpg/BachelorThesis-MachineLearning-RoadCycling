"""Compare logged experiment runs from the terminal.

    python -m cyclingml.runs                       # list experiments
    python -m cyclingml.runs profiling             # all runs of one experiment, best first
    python -m cyclingml.runs profiling --metric silhouette --top 10

Used by the /results skill so Claude and Kevin read the same table.
"""

from __future__ import annotations

import argparse

import pandas as pd


def experiments() -> pd.DataFrame:
    import mlflow

    from cyclingml.tracking import tracking_uri

    mlflow.set_tracking_uri(tracking_uri())
    rows = [
        {"experiment": e.name, "id": e.experiment_id}
        for e in mlflow.search_experiments()
        if e.name != "Default"
    ]
    return pd.DataFrame(rows)


def runs_table(
    experiment: str, metric: str | None = None, top: int = 20, ascending: bool = False
) -> pd.DataFrame:
    """One row per run: name, start time, git commit, params and metrics, sorted by `metric`."""
    import mlflow

    from cyclingml.tracking import tracking_uri

    mlflow.set_tracking_uri(tracking_uri())
    df = mlflow.search_runs(experiment_names=[experiment])
    if df.empty:
        return df
    keep = ["tags.mlflow.runName", "start_time", "tags.git_commit", "tags.config_hash"]
    params = sorted(c for c in df.columns if c.startswith("params."))
    metrics = sorted(c for c in df.columns if c.startswith("metrics."))
    df = df[[c for c in keep if c in df.columns] + params + metrics]
    df.columns = [c.split(".", 1)[1] if "." in c else c for c in df.columns]
    df = df.rename(columns={"mlflow.runName": "run"})
    if metric and metric in df.columns:
        df = df.sort_values(metric, ascending=ascending)
    else:
        df = df.sort_values("start_time", ascending=False)
    return df.head(top)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Compare MLflow runs")
    p.add_argument("experiment", nargs="?")
    p.add_argument("--metric", help="sort by this metric (best first)")
    p.add_argument("--ascending", action="store_true", help="lower is better (e.g. MAE)")
    p.add_argument("--top", type=int, default=20)
    args = p.parse_args(argv)
    pd.set_option("display.width", 200, "display.max_columns", 50)
    if not args.experiment:
        print(experiments().to_string(index=False))
    else:
        table = runs_table(args.experiment, args.metric, args.top, args.ascending)
        print(table.to_string(index=False) if len(table) else "no runs")


if __name__ == "__main__":
    main()
