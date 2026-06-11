"""Unit tests for analysis.decompile (phase 5): exact piecewise-affine trees."""

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from kill_nonlinearities.analysis.decompile import (
    Branch,
    Leaf,
    Truncated,
    decompile_mlp,
    evaluate_tree,
    render_tree,
    tree_stats,
)
from kill_nonlinearities.config import ModelConfig
from kill_nonlinearities.models.activations import ActivationMode
from kill_nonlinearities.models.mlp import ReLUMLP

INPUT_DIM = 4


def _model(
    hidden: tuple[int, ...], seed: int, modes: list[list[int]] | None = None
) -> ReLUMLP:
    torch.manual_seed(seed)
    model = ReLUMLP(
        ModelConfig(input_dim=INPUT_DIM, hidden_dims=hidden, output_dim=3)
    ).double()
    if modes is not None:
        for act, site_modes in zip(model.activations, modes, strict=True):
            act.set_modes(torch.tensor(site_modes, dtype=torch.int64))
    return model


def _box(seed: int, width: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    lo = torch.randn(INPUT_DIM, generator=gen, dtype=torch.float64)
    return lo, lo + width * torch.rand(INPUT_DIM, generator=gen, dtype=torch.float64)


def _samples(lo: torch.Tensor, hi: torch.Tensor, n: int, seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return lo + torch.rand(n, INPUT_DIM, generator=gen, dtype=torch.float64) * (hi - lo)


@settings(max_examples=15, deadline=None)
@given(
    hidden=st.lists(st.integers(min_value=1, max_value=4), min_size=1, max_size=2),
    seed=st.integers(min_value=0, max_value=10_000),
)
def test_tree_evaluates_exactly_like_the_network(hidden: list[int], seed: int) -> None:
    """Property: the decompiled program equals the network on the box."""
    rng = torch.Generator().manual_seed(seed)
    modes = [torch.randint(0, 3, (w,), generator=rng).tolist() for w in hidden]
    model = _model(tuple(hidden), seed=seed % 1000, modes=modes)
    lo, hi = _box(seed % 997, width=2.0)
    tree = decompile_mlp(model, lo, hi, max_leaves=512)
    assert tree_stats(tree).n_truncated == 0
    x = _samples(lo, hi, n=64, seed=seed % 991)
    model.eval()
    with torch.no_grad():
        expected = model(x).logits
    torch.testing.assert_close(evaluate_tree(tree, x), expected, rtol=0, atol=1e-9)


def test_fully_stable_network_is_a_single_leaf() -> None:
    """Huge biases make every unit stable: the program is one affine map."""
    model = _model((3, 3), seed=5)
    with torch.no_grad():
        model.linears[0].bias.copy_(torch.tensor([50.0, -50.0, 50.0]).double())
        model.linears[1].bias.copy_(torch.tensor([-50.0, 50.0, 50.0]).double())
    lo, hi = _box(7, width=0.5)
    tree = decompile_mlp(model, lo, hi)
    assert isinstance(tree, Leaf)
    x = _samples(lo, hi, n=16, seed=11)
    model.eval()
    with torch.no_grad():
        torch.testing.assert_close(
            evaluate_tree(tree, x), model(x).logits, rtol=0, atol=1e-9
        )


def test_single_unstable_unit_yields_one_branch_two_leaves() -> None:
    model = _model((1,), seed=13)
    with torch.no_grad():
        model.linears[0].bias.zero_()  # the lone unit straddles 0 on a wide box
        model.linears[0].weight.copy_(torch.ones(1, INPUT_DIM).double())
    lo = -torch.ones(INPUT_DIM, dtype=torch.float64)
    hi = torch.ones(INPUT_DIM, dtype=torch.float64)
    tree = decompile_mlp(model, lo, hi)
    stats = tree_stats(tree)
    assert isinstance(tree, Branch)
    assert (stats.n_branches, stats.n_leaves, stats.n_truncated) == (1, 2, 0)
    assert tree.site == "relu0" and tree.index == 0


def test_branch_routing_matches_the_hyperplane_test() -> None:
    """A sample on the positive side of the root test routes to ``high``."""
    model = _model((1,), seed=13)
    with torch.no_grad():
        model.linears[0].bias.zero_()
        model.linears[0].weight.copy_(torch.ones(1, INPUT_DIM).double())
    lo = -torch.ones(INPUT_DIM, dtype=torch.float64)
    hi = torch.ones(INPUT_DIM, dtype=torch.float64)
    tree = decompile_mlp(model, lo, hi)
    assert isinstance(tree, Branch)
    positive = torch.full((1, INPUT_DIM), 0.5, dtype=torch.float64)
    negative = -positive
    assert isinstance(tree.high, Leaf) and isinstance(tree.low, Leaf)
    torch.testing.assert_close(
        evaluate_tree(tree, positive), evaluate_tree(tree.high, positive)
    )
    torch.testing.assert_close(
        evaluate_tree(tree, negative), evaluate_tree(tree.low, negative)
    )


def test_budget_exhaustion_produces_truncated_nodes() -> None:
    model = _model((4, 4), seed=17)
    lo, hi = _box(19, width=4.0)
    tree = decompile_mlp(model, lo, hi, max_leaves=1)
    stats = tree_stats(tree)
    assert stats.n_truncated >= 1
    with pytest.raises(ValueError, match="truncated"):
        evaluate_tree(Truncated(), torch.zeros(1, INPUT_DIM))


def test_zero_and_identity_modes_do_not_branch() -> None:
    """Masked units are exact affine behavior — never branch points."""
    model = _model(
        (3,),
        seed=23,
        modes=[
            [
                int(ActivationMode.ZERO),
                int(ActivationMode.IDENTITY),
                int(ActivationMode.ZERO),
            ]
        ],
    )
    lo, hi = _box(29, width=4.0)
    tree = decompile_mlp(model, lo, hi)
    assert isinstance(tree, Leaf)
    x = _samples(lo, hi, n=16, seed=31)
    model.eval()
    with torch.no_grad():
        torch.testing.assert_close(
            evaluate_tree(tree, x), model(x).logits, rtol=0, atol=1e-9
        )


def test_render_tree_emits_readable_conditionals() -> None:
    model = _model((1,), seed=13)
    with torch.no_grad():
        model.linears[0].bias.zero_()
        model.linears[0].weight.copy_(torch.ones(1, INPUT_DIM).double())
    lo = -torch.ones(INPUT_DIM, dtype=torch.float64)
    hi = torch.ones(INPUT_DIM, dtype=torch.float64)
    text = render_tree(decompile_mlp(model, lo, hi))
    assert "if " in text and "else:" in text
    assert "relu0[0] fires" in text
    assert "return affine" in text


def test_invalid_inputs_are_rejected() -> None:
    model = _model((2,), seed=37)
    lo = torch.zeros(INPUT_DIM, dtype=torch.float64)
    with pytest.raises(ValueError, match="lower > upper"):
        decompile_mlp(model, lo, lo - 1.0)
    with pytest.raises(ValueError, match="max_leaves"):
        decompile_mlp(model, lo, lo + 1.0, max_leaves=0)
