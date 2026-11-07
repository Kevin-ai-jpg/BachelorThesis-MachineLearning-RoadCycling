import importlib

import pytest

import cyclingml
from cyclingml import paths
from cyclingml.config import config_hash, flatten, load_config


def test_version():
    assert cyclingml.__version__


@pytest.mark.parametrize("sub", ["io", "clean", "features", "models", "eval"])
def test_subpackages_import(sub):
    importlib.import_module(f"cyclingml.{sub}")


def test_paths_follow_env(tmp_path, monkeypatch):
    monkeypatch.setenv("CYCLINGML_DATA", str(tmp_path))
    monkeypatch.delenv("CYCLINGML_MLRUNS", raising=False)
    assert paths.data_root() == tmp_path
    assert paths.mlruns_dir() == tmp_path / "mlruns"
    paths.ensure_dirs()
    assert (tmp_path / "interim").is_dir()


def test_config_roundtrip(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("model:\n  name: kmeans\n  k: 5\nseed: 1\n")
    cfg = load_config(p)
    assert flatten(cfg) == {"model.name": "kmeans", "model.k": 5, "seed": 1}
    assert config_hash(cfg) == config_hash({"seed": 1, "model": {"k": 5, "name": "kmeans"}})
    assert config_hash(cfg) != config_hash({**cfg, "seed": 2})


def test_config_rejects_list(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("- a\n- b\n")
    with pytest.raises(ValueError):
        load_config(p)
