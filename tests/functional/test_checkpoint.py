"""Functional tests for checkpoint save/load round-trip (spec §4.7)."""

import dataclasses
from pathlib import Path

import torch

from kill_nonlinearities.config import ExperimentConfig, ModelConfig
from kill_nonlinearities.models.activations import ActivationMode
from kill_nonlinearities.models.mlp import ReLUMLP
from kill_nonlinearities.training.checkpoint import (
    load_checkpoint,
    save_checkpoint,
)


def _make_config() -> ExperimentConfig:
    cfg = ExperimentConfig(name="ckpt-test")
    return dataclasses.replace(
        cfg, model=ModelConfig(input_dim=8, hidden_dims=(6, 4), output_dim=3)
    )


def test_round_trip_reproduces_logits_step_and_config(tmp_path: Path) -> None:
    torch.manual_seed(0)
    config = _make_config()
    model = ReLUMLP(config.model)
    x = torch.randn(5, 8)
    expected = model(x).logits

    path = save_checkpoint(model, step=7, config=config, dir=tmp_path)
    assert path.exists()

    # Strict load into an architecturally-identical fresh model.
    fresh = ReLUMLP(config.model)
    step = load_checkpoint(path, fresh)

    assert step == 7
    assert torch.equal(fresh(x).logits, expected)


def test_round_trip_preserves_non_default_mode_buffer(tmp_path: Path) -> None:
    torch.manual_seed(1)
    config = _make_config()
    model = ReLUMLP(config.model)
    # Non-default mode on the first SelectiveReLU site (hidden_dims[0] == 6).
    modes = torch.tensor(
        [
            int(ActivationMode.RELU),
            int(ActivationMode.ZERO),
            int(ActivationMode.IDENTITY),
            int(ActivationMode.ZERO),
            int(ActivationMode.IDENTITY),
            int(ActivationMode.RELU),
        ],
        dtype=torch.int64,
    )
    model.activations[0].set_modes(modes)
    x = torch.randn(4, 8)
    expected = model(x).logits

    path = save_checkpoint(model, step=2, config=config, dir=tmp_path)
    fresh = ReLUMLP(config.model)
    step = load_checkpoint(path, fresh)

    assert step == 2
    assert torch.equal(fresh.activations[0].mode, modes)
    assert torch.equal(fresh(x).logits, expected)


def test_saved_config_deserializes_equal_to_original(tmp_path: Path) -> None:
    config = _make_config()
    model = ReLUMLP(config.model)
    path = save_checkpoint(model, step=0, config=config, dir=tmp_path)

    blob = torch.load(path, weights_only=False)
    assert blob["config"] == config
    assert blob["step"] == 0
