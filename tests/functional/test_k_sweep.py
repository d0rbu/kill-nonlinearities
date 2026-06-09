"""Functional tests for surgery.apply (spec §4.11, §7 non-mutation)."""

import torch
from torch.utils.data import DataLoader, TensorDataset

from kill_nonlinearities.config import ModelConfig
from kill_nonlinearities.models.activations import ActivationMode
from kill_nonlinearities.models.mlp import ReLUMLP
from kill_nonlinearities.surgery.apply import (
    apply_modes,
    evaluate_accuracy,
)


def _labeled_loader(n: int, input_dim: int, classes: int) -> DataLoader:
    x = torch.randn(n, input_dim)
    y = torch.randint(0, classes, (n,))
    return DataLoader(TensorDataset(x, y), batch_size=4, shuffle=False)


def test_apply_modes_sets_buffers() -> None:
    torch.manual_seed(0)
    model = ReLUMLP(ModelConfig(input_dim=4, hidden_dims=(3,), output_dim=2))
    site = model.site_names[0]
    modes = torch.tensor(
        [
            int(ActivationMode.ZERO),
            int(ActivationMode.IDENTITY),
            int(ActivationMode.RELU),
        ],
        dtype=torch.int64,
    )
    apply_modes(model, {site: modes})
    torch.testing.assert_close(model.activations[0].mode, modes, rtol=0, atol=0)


def test_evaluate_accuracy_perfect_and_chance() -> None:
    torch.manual_seed(0)
    model = ReLUMLP(ModelConfig(input_dim=4, hidden_dims=(3,), output_dim=2))
    loader = _labeled_loader(8, input_dim=4, classes=2)
    acc = evaluate_accuracy(model, loader, device="cpu")
    assert 0.0 <= acc <= 1.0
