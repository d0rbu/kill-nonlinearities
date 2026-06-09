"""Unit tests for the PURE sweep helpers (spec §4.13, §7).

`build_sweep_config` and `config_from_wandb` are pure: no wandb, no network, no torch RNG.
"""

from dataclasses import replace

import pytest

from kill_nonlinearities.config import (
    DataConfig,
    ExperimentConfig,
    OptimConfig,
    RegConfig,
    TrainConfig,
)
from kill_nonlinearities.experiments.sweep import (
    BASE_CONFIG,
    build_sweep_config,
    config_from_wandb,
)


def test_build_sweep_config_exact_dict() -> None:
    """grids + method → the exact wandb sweep dict, incl. the frozen metric block ([R20])."""
    grids: dict[str, list[object]] = {
        "lr": [1e-3, 1e-2],
        "epochs": [10, 20],
        "batch_size": [64, 128],
        "lam": [0.0, 0.1],
    }

    result = build_sweep_config(grids, "grid")

    assert result == {
        "method": "grid",
        "metric": {"name": "val/acc", "goal": "maximize"},
        "parameters": {
            "lr": {"values": [1e-3, 1e-2]},
            "epochs": {"values": [10, 20]},
            "batch_size": {"values": [64, 128]},
            "lam": {"values": [0.0, 0.1]},
        },
    }


def test_build_sweep_config_passes_method_through() -> None:
    """The `method` argument is copied verbatim into the config."""
    result = build_sweep_config({"lr": [1e-3]}, "bayes")

    assert result["method"] == "bayes"
    assert result["metric"] == {"name": "val/acc", "goal": "maximize"}
    assert result["parameters"] == {"lr": {"values": [1e-3]}}


def test_config_from_wandb_full_nested_equality() -> None:
    """Flat swept dict → the exact nested ExperimentConfig; non-swept fields from BASE ([R9])."""
    mapping = {"lr": 1e-2, "epochs": 5, "batch_size": 64, "lam": 0.1}

    cfg = config_from_wandb(mapping)

    expected = replace(
        BASE_CONFIG,
        optim=replace(BASE_CONFIG.optim, lr=1e-2),
        train=replace(BASE_CONFIG.train, epochs=5),
        data=replace(BASE_CONFIG.data, batch_size=64),
        reg=replace(BASE_CONFIG.reg, lam=0.1),
    )

    assert cfg == expected


def test_config_from_wandb_only_maps_swept_fields() -> None:
    """Only lr/epochs/batch_size/lam are overridden; every other field equals BASE ([R9])."""
    cfg = config_from_wandb({"lr": 1e-2, "epochs": 5, "batch_size": 64, "lam": 0.1})

    assert cfg.optim == OptimConfig(lr=1e-2)
    assert cfg.train == TrainConfig(epochs=5)
    assert cfg.data == DataConfig(batch_size=64)
    assert cfg.reg == RegConfig(lam=0.1)
    # Untouched nested configs are exactly the BASE ones.
    assert cfg.model == BASE_CONFIG.model
    assert cfg.temp_schedule == BASE_CONFIG.temp_schedule
    assert cfg.probe == BASE_CONFIG.probe
    assert cfg.checkpoint == BASE_CONFIG.checkpoint
    assert cfg.surgery == BASE_CONFIG.surgery
    assert cfg.wandb == BASE_CONFIG.wandb
    assert cfg.name == BASE_CONFIG.name


def test_base_config_is_experiment_config() -> None:
    """BASE_CONFIG is a fully-formed ExperimentConfig used as the documented default ([R9])."""
    assert isinstance(BASE_CONFIG, ExperimentConfig)


def test_config_from_wandb_unknown_key_raises() -> None:
    """An unrecognized flat key raises KeyError (no silent drop) ([R9])."""
    mapping = {"lr": 1e-2, "epochs": 5, "batch_size": 64, "lam": 0.1, "bogus": 1}

    with pytest.raises(KeyError, match=r"unknown"):
        config_from_wandb(mapping)


def test_config_from_wandb_missing_swept_key_raises() -> None:
    """A missing swept key (here: lam) raises ([R9])."""
    mapping = {"lr": 1e-2, "epochs": 5, "batch_size": 64}

    with pytest.raises(KeyError, match=r"missing"):
        config_from_wandb(mapping)


def test_config_from_wandb_non_numeric_lr_raises() -> None:
    """A non-numeric `lr` is coerced at the boundary and raises ValueError ([R9])."""
    mapping = {"lr": "oops", "epochs": 5, "batch_size": 64, "lam": 0.1}

    with pytest.raises(ValueError):
        config_from_wandb(mapping)


def test_config_from_wandb_non_numeric_batch_size_raises() -> None:
    """A clearly-invalid `batch_size` is coerced at the boundary and raises ValueError ([R9])."""
    mapping = {"lr": 1e-2, "epochs": 5, "batch_size": "abc", "lam": 0.1}

    with pytest.raises(ValueError):
        config_from_wandb(mapping)
