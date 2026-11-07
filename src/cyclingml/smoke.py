"""Setup check: logs one dummy MLflow run.

python -m cyclingml.smoke configs/smoke.yaml
"""

from __future__ import annotations

import sys

import numpy as np

from cyclingml.config import load_config
from cyclingml.tracking import run


def main(config_path: str) -> str:
    cfg = load_config(config_path)
    rng = np.random.default_rng(cfg.get("seed", 0))
    with run(cfg.get("experiment", "smoke-test"), cfg, run_name="smoke") as r:
        r.log_metrics({"random_score": float(rng.random())})
        print(f"Logged run {r.id}")
        return r.id


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "configs/smoke.yaml")
