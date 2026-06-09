"""Functional tests for surgery.apply (spec §4.11, §7 non-mutation)."""

import torch
from torch.utils.data import DataLoader, TensorDataset

from kill_nonlinearities.analysis.selection import make_k_grid, rank_by_entropy
from kill_nonlinearities.analysis.statistics import (
    collect_pre_activations,
    neuron_stats,
)
from kill_nonlinearities.config import ModelConfig
from kill_nonlinearities.models.activations import ActivationMode
from kill_nonlinearities.models.mlp import ReLUMLP
from kill_nonlinearities.surgery.apply import (
    KPoint,
    apply_modes,
    evaluate_accuracy,
    k_sweep,
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


def test_k_sweep_returns_point_per_grid_value() -> None:
    torch.manual_seed(0)
    model = ReLUMLP(ModelConfig(input_dim=4, hidden_dims=(3, 2), output_dim=2))
    val = _labeled_loader(8, input_dim=4, classes=2)
    test = _labeled_loader(8, input_dim=4, classes=2)

    pre = collect_pre_activations(model, val, device="cpu")
    ranked = rank_by_entropy(neuron_stats(pre))
    total = len(ranked)  # 3 + 2 = 5 neurons
    grid = make_k_grid(total, num_k=21)

    points = k_sweep(model, ranked, grid, val, test, device="cpu")

    assert [p.k for p in points] == grid
    assert all(isinstance(p, KPoint) for p in points)
    assert all(0.0 <= p.val_acc <= 1.0 and 0.0 <= p.test_acc <= 1.0 for p in points)


def test_k_sweep_does_not_mutate_canonical_model() -> None:
    torch.manual_seed(0)
    model = ReLUMLP(ModelConfig(input_dim=4, hidden_dims=(3, 2), output_dim=2))
    val = _labeled_loader(8, input_dim=4, classes=2)
    test = _labeled_loader(8, input_dim=4, classes=2)

    pre = collect_pre_activations(model, val, device="cpu")
    ranked = rank_by_entropy(neuron_stats(pre))
    grid = make_k_grid(len(ranked), num_k=21)

    # Snapshot canonical modes (all RELU) and logits BEFORE the sweep.
    mode_snapshot = {
        site: act.mode.clone()
        for site, act in zip(model.site_names, model.activations, strict=True)
    }
    probe = torch.randn(5, 4)
    model.eval()
    with torch.no_grad():
        logits_before = model(probe).logits.clone()

    k_sweep(model, ranked, grid, val, test, device="cpu")

    # Modes still all-RELU; logits bit-for-bit unchanged.
    for site, act in zip(model.site_names, model.activations, strict=True):
        assert torch.equal(act.mode, mode_snapshot[site])
        assert torch.equal(act.mode, torch.zeros_like(act.mode))  # RELU == 0
    with torch.no_grad():
        logits_after = model(probe).logits
    assert torch.equal(logits_after, logits_before)
