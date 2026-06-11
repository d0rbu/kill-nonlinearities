"""Unit tests for surgery.fold (phase 2): fold/trim preserve the masked function."""

import copy

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from kill_nonlinearities.config import ModelConfig
from kill_nonlinearities.models.activations import ActivationMode
from kill_nonlinearities.models.mlp import ReLUMLP
from kill_nonlinearities.surgery.fold import (
    FoldedMLP,
    fold_mlp,
    folded_stats,
    trim_folded,
)

RELU = int(ActivationMode.RELU)
ZERO = int(ActivationMode.ZERO)
IDENT = int(ActivationMode.IDENTITY)


def _model(
    hidden_dims: tuple[int, ...], modes_per_site: list[list[int]], seed: int = 0
) -> ReLUMLP:
    """A seeded ReLUMLP with the given per-site mode assignments applied."""
    torch.manual_seed(seed)
    model = ReLUMLP(ModelConfig(input_dim=7, hidden_dims=hidden_dims, output_dim=4))
    for act, modes in zip(model.activations, modes_per_site, strict=True):
        act.set_modes(torch.tensor(modes, dtype=torch.int64))
    return model


def _masked_logits(model: ReLUMLP, x: torch.Tensor) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        return model(x).logits


def _assert_equivalent(model: ReLUMLP, folded: FoldedMLP, seed: int = 1) -> None:
    """Folded logits match the masked model: float32 close AND float64 tight.

    The float64 leg re-folds a float64 copy of the model (the weight products
    must be COMPOSED in float64 — folding in float32 and casting after would
    bake float32 rounding into the composed weights).
    """
    torch.manual_seed(seed)
    x = torch.randn(16, 7)
    with torch.no_grad():
        torch.testing.assert_close(folded(x), _masked_logits(model, x))
        model64 = copy.deepcopy(model).double()
        folded64 = trim_folded(fold_mlp(model64))
        torch.testing.assert_close(
            folded64(x.double()),
            model64(x.double()).logits,
            rtol=0,
            atol=1e-10,
        )


def test_all_relu_fold_matches_original() -> None:
    """With every neuron RELU the fold reproduces the untouched model."""
    model = _model((5, 6), [[RELU] * 5, [RELU] * 6])
    folded = fold_mlp(model)
    _assert_equivalent(model, folded)
    stats = folded_stats(folded)
    assert stats.nonlinear_widths == (5, 6)


def test_all_identity_collapses_to_single_affine() -> None:
    """A fully-IDENTITY network folds into one affine map (zero ReLU units)."""
    model = _model((5, 6), [[IDENT] * 5, [IDENT] * 6])
    folded = trim_folded(fold_mlp(model))
    _assert_equivalent(model, folded)
    assert folded_stats(folded).nonlinear_widths == (0, 0)

    # The head equals the explicit composed affine map.
    w0, w1 = model.linears[0], model.linears[1]
    with torch.no_grad():
        composed_w = model.head.weight @ w1.weight @ w0.weight
        composed_b = (
            model.head.weight @ (w1.weight @ w0.bias + w1.bias) + model.head.bias
        )
        torch.testing.assert_close(folded.head.weight, composed_w)
        torch.testing.assert_close(folded.head.bias, composed_b)


def test_all_zero_yields_constant_logits() -> None:
    """A fully-ZERO network computes input-independent logits (bias chain)."""
    model = _model((5, 6), [[ZERO] * 5, [ZERO] * 6])
    folded = trim_folded(fold_mlp(model))
    _assert_equivalent(model, folded)
    x = torch.randn(8, 7)
    with torch.no_grad():
        logits = folded(x)
    assert torch.equal(logits, logits[0].expand_as(logits))


def test_trim_removes_carry_when_no_identity_neurons() -> None:
    """RELU/ZERO-only models trim to a plain smaller MLP (empty carries)."""
    model = _model(
        (6, 5), [[RELU, ZERO, RELU, ZERO, ZERO, RELU], [ZERO] * 2 + [RELU] * 3]
    )
    folded = trim_folded(fold_mlp(model))
    _assert_equivalent(model, folded)
    stats = folded_stats(folded)
    assert stats.nonlinear_widths == (3, 3)
    assert stats.carry_widths == (0, 0)


def test_trim_drops_relu_unit_with_zero_downstream_column() -> None:
    """A kept RELU unit that no consumer reads is pruned by the trim."""
    model = _model((4,), [[RELU] * 4])
    with torch.no_grad():
        model.head.weight[:, 2] = 0.0  # nobody reads hidden unit 2
    folded = trim_folded(fold_mlp(model))
    _assert_equivalent(model, folded)
    assert folded_stats(folded).nonlinear_widths == (3,)


@settings(max_examples=25, deadline=None)
@given(
    hidden=st.lists(st.integers(min_value=1, max_value=6), min_size=1, max_size=3),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_fold_and_trim_preserve_function_under_random_modes(
    hidden: list[int], seed: int
) -> None:
    """Property: any mode assignment folds (and trims) to the same function."""
    rng = torch.Generator().manual_seed(seed)
    modes = [torch.randint(0, 3, (width,), generator=rng).tolist() for width in hidden]
    model = _model(tuple(hidden), modes, seed=seed % 1000)
    folded = fold_mlp(model)
    _assert_equivalent(model, folded, seed=seed % 997)
    trimmed = trim_folded(folded)
    _assert_equivalent(model, trimmed, seed=seed % 991)
    # Trimming never grows anything.
    before, after = folded_stats(folded), folded_stats(trimmed)
    assert after.params <= before.params
    assert all(
        a <= b
        for a, b in zip(after.nonlinear_widths, before.nonlinear_widths, strict=True)
    )


def test_folded_stats_counts_match_structure() -> None:
    model = _model((5,), [[RELU, RELU, IDENT, ZERO, ZERO]])
    folded = fold_mlp(model)
    stats = folded_stats(folded)
    assert stats.nonlinear_widths == (2,)
    assert stats.carry_widths == (7,)  # untrimmed: full input carried
    # Linear(7->2) + head Linear(2+7 -> 4) parameters.
    assert stats.params == (2 * 7 + 2) + (4 * 9 + 4)


def test_folded_mlp_rejects_mismatched_carries() -> None:
    stage = torch.nn.Linear(3, 2)
    head = torch.nn.Linear(5, 2)
    with pytest.raises(ValueError, match="carry"):
        FoldedMLP([stage], [], head)


def test_folded_mlp_rejects_non_int64_carry() -> None:
    stage = torch.nn.Linear(3, 2)
    head = torch.nn.Linear(5, 2)
    with pytest.raises(ValueError, match="int64"):
        FoldedMLP([stage], [torch.arange(3, dtype=torch.int32)], head)


def test_folded_mlp_state_dict_round_trips() -> None:
    model = _model(
        (5, 4), [[RELU, IDENT, ZERO, RELU, IDENT], [IDENT, RELU, ZERO, RELU]]
    )
    folded = trim_folded(fold_mlp(model))
    clone = copy.deepcopy(folded)
    clone.load_state_dict(folded.state_dict())
    x = torch.randn(4, 7)
    with torch.no_grad():
        torch.testing.assert_close(clone(x), folded(x), rtol=0, atol=0)
