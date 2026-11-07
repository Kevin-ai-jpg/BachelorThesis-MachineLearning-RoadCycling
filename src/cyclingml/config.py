"""Experiment configs: one YAML file per experiment in configs/."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"{path}: top level of a config must be a mapping")
    return cfg


def flatten(cfg: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Nested dict -> {"model.k": 5, ...}, the form MLflow stores as params."""
    out: dict[str, Any] = {}
    for key, value in cfg.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(flatten(value, f"{name}."))
        else:
            out[name] = value
    return out


def config_hash(cfg: dict[str, Any]) -> str:
    """Short stable id for a config, so identical settings are easy to spot across runs."""
    blob = json.dumps(cfg, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:10]
