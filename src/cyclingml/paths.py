"""Where data and results live.

Every path comes from an environment variable so the same code runs on the laptop,
on Google Colab (Drive mounted) and in CI without edits.

    CYCLINGML_DATA    root data folder   (default: <repo>/data)
    CYCLINGML_MLRUNS  MLflow store       (default: <data root>/mlruns)
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def data_root() -> Path:
    return Path(os.environ.get("CYCLINGML_DATA", REPO_ROOT / "data")).expanduser()


def raw_dir(source: str) -> Path:
    """Raw downloads for one source, e.g. raw_dir("goldencheetah")."""
    return data_root() / "raw" / source


def interim_dir() -> Path:
    """Cleaned 1 Hz Parquet rides."""
    return data_root() / "interim"


def processed_dir() -> Path:
    """Feature tables ready for modelling."""
    return data_root() / "processed"


def reference_dir() -> Path:
    """Small, versioned reference tables (pro power profiles). Lives in the repo."""
    return REPO_ROOT / "reference"


def mlruns_dir() -> Path:
    return Path(os.environ.get("CYCLINGML_MLRUNS", data_root() / "mlruns")).expanduser()


def ensure_dirs() -> None:
    for p in (data_root(), interim_dir(), processed_dir(), mlruns_dir()):
        p.mkdir(parents=True, exist_ok=True)
