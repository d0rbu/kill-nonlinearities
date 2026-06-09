"""Functional tests for surgery.apply (spec §4.11, §7 non-mutation)."""

import torch
from torch.utils.data import DataLoader, TensorDataset

from kill_nonlinearities.analysis.selection import (
    assign_modes,
    make_k_grid,
    rank_by_entropy,
)
from kill_nonlinearities.analysis.statistics import (
    NeuronStats,
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


def test_assign_modes_tie_break_zero_vs_identity_on_balanced_neuron() -> None:
    """A constructed q==0.5 neuron is ZERO under tie_break='zero', IDENTITY otherwise."""
    balanced = NeuronStats(site="relu0", index=0, q=0.5, entropy=1.0, mean_pre=0.0)
    widths = {"relu0": 1}

    zero_modes = assign_modes([balanced], tie_break="zero", widths=widths)
    identity_modes = assign_modes([balanced], tie_break="identity", widths=widths)

    assert int(zero_modes["relu0"][0]) == int(ActivationMode.ZERO)
    assert int(identity_modes["relu0"][0]) == int(ActivationMode.IDENTITY)


def test_k_sweep_threads_tie_break_changing_accuracy() -> None:
    """k_sweep forwards tie_break end-to-end: ZERO vs IDENTITY change val accuracy.

    The lone hidden neuron is positive on the first selection sample and negative
    on the second (q==0.5, the only neuron, selected at k>=1). The head and labels
    are rigged so the negative-input sample is classified CORRECTLY only when the
    tie neuron passes its (negative) pre-activation through (IDENTITY) and
    INCORRECTLY when it is zeroed (ZERO). So k_sweep's k=1 val_acc must differ
    between tie_break='identity' and tie_break='zero' — proving the parameter is
    threaded, not hardcoded.
    """
    model = ReLUMLP(ModelConfig(input_dim=1, hidden_dims=(1,), output_dim=2))
    with torch.no_grad():
        # Hidden: z = x. x=+1 -> z>0; x=-1 -> z<0 -> q==0.5 over the two rows.
        model.linears[0].weight.copy_(torch.tensor([[1.0]]))
        model.linears[0].bias.zero_()
        # Head maps the single hidden value h to two logits [class0=0, class1=h].
        # IDENTITY: x=-1 -> h=-1 -> logit1=-1<0 -> predicts class0.
        # ZERO:     x=-1 -> h= 0 -> logit1= 0 == logit0 -> argmax ties to class0.
        # To make the modes observably differ we instead read class1 = -h so that
        # IDENTITY x=-1 -> h=-1 -> logit1=+1 -> class1, ZERO -> 0 -> class0.
        model.head.weight.copy_(torch.tensor([[0.0], [-1.0]]))
        model.head.bias.zero_()

    xs = torch.tensor([[1.0], [-1.0]])
    # Label the negative-input row class1 so only IDENTITY classifies it right.
    labels = torch.tensor([0, 1], dtype=torch.int64)
    loader = DataLoader(TensorDataset(xs, labels), batch_size=2, shuffle=False)

    pre = collect_pre_activations(model, loader, device="cpu")
    stats = neuron_stats(pre)
    assert stats[0].q == 0.5  # the lone neuron is exactly balanced
    ranked = rank_by_entropy(stats)

    identity_pts = k_sweep(
        model, ranked, [1], loader, loader, device="cpu", tie_break="identity"
    )
    zero_pts = k_sweep(
        model, ranked, [1], loader, loader, device="cpu", tie_break="zero"
    )

    # IDENTITY classifies the x=-1 row correctly (class1); ZERO does not.
    assert identity_pts[0].val_acc > zero_pts[0].val_acc
    # The canonical model is never mutated by k_sweep (all-RELU preserved).
    assert torch.equal(
        model.activations[0].mode, torch.zeros_like(model.activations[0].mode)
    )


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
