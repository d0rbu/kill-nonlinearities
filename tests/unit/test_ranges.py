"""Unit tests for analysis.ranges (phase 4): sound, ordered, mode-aware bounds."""

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from torch.utils.data import DataLoader, TensorDataset

from kill_nonlinearities.analysis.ranges import (
    certified_modes,
    data_box,
    interval_ranges,
    lp_ranges,
)
from kill_nonlinearities.config import ModelConfig
from kill_nonlinearities.models.activations import ActivationMode
from kill_nonlinearities.models.mlp import ReLUMLP

RELU = int(ActivationMode.RELU)
ZERO = int(ActivationMode.ZERO)
IDENT = int(ActivationMode.IDENTITY)

INPUT_DIM = 5


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


def _box(seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    lo = torch.randn(INPUT_DIM, generator=gen, dtype=torch.float64)
    width = torch.rand(INPUT_DIM, generator=gen, dtype=torch.float64)
    return lo, lo + width


def _sampled_pre_acts(
    model: ReLUMLP, lo: torch.Tensor, hi: torch.Tensor, n: int, seed: int
) -> list[torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    x = lo + torch.rand(n, INPUT_DIM, generator=gen, dtype=torch.float64) * (hi - lo)
    model.eval()
    with torch.no_grad():
        return list(model(x).pre_activations)


@settings(max_examples=12, deadline=None)
@given(
    hidden=st.lists(st.integers(min_value=1, max_value=4), min_size=1, max_size=2),
    seed=st.integers(min_value=0, max_value=10_000),
)
def test_bounds_are_sound_and_lp_is_tighter(hidden: list[int], seed: int) -> None:
    """Property: sampled pre-acts lie inside IBP and LP bounds; LP ⊆ IBP."""
    rng = torch.Generator().manual_seed(seed)
    modes = [torch.randint(0, 3, (w,), generator=rng).tolist() for w in hidden]
    model = _model(tuple(hidden), seed=seed % 1000, modes=modes)
    lo, hi = _box(seed % 997)
    ibp = interval_ranges(model, lo, hi)
    lp = lp_ranges(model, lo, hi)
    samples = _sampled_pre_acts(model, lo, hi, n=64, seed=seed % 991)
    for ibp_layer, lp_layer, z in zip(ibp, lp, samples, strict=True):
        for layer in (ibp_layer, lp_layer):
            assert bool((z >= layer.pre_lower - 1e-6).all()), layer.site
            assert bool((z <= layer.pre_upper + 1e-6).all()), layer.site
        assert bool((lp_layer.pre_lower >= ibp_layer.pre_lower - 1e-5).all())
        assert bool((lp_layer.pre_upper <= ibp_layer.pre_upper + 1e-5).all())


def test_first_layer_lp_equals_interval_closed_form() -> None:
    """Over a box, the layer-1 LP optimum is exactly the IBP closed form."""
    model = _model((4, 3), seed=3)
    lo, hi = _box(7)
    ibp = interval_ranges(model, lo, hi)
    lp = lp_ranges(model, lo, hi)
    torch.testing.assert_close(lp[0].pre_lower, ibp[0].pre_lower, rtol=1e-6, atol=1e-5)
    torch.testing.assert_close(lp[0].pre_upper, ibp[0].pre_upper, rtol=1e-6, atol=1e-5)


def test_point_box_recovers_exact_pre_activations() -> None:
    """A degenerate box (lower == upper) gives exact bounds in both methods."""
    model = _model((4, 3), seed=5)
    point = torch.randn(INPUT_DIM, dtype=torch.float64)
    model.eval()
    with torch.no_grad():
        exact = list(model(point.unsqueeze(0)).pre_activations)
    for method in (interval_ranges, lp_ranges):
        for layer, z in zip(method(model, point, point), exact, strict=True):
            torch.testing.assert_close(layer.pre_lower, z[0], rtol=1e-6, atol=1e-5)
            torch.testing.assert_close(layer.pre_upper, z[0], rtol=1e-6, atol=1e-5)


def test_zero_mode_equals_cutting_the_units_fan_out() -> None:
    """ZERO-pinning a unit gives the same downstream bounds as zeroing its fan-out.

    (Note ZERO-pinning need NOT narrow bounds versus the plain model: zeroing a
    stably-POSITIVE unit moves its activation outside the plain range. The
    correct algebraic property is fan-out equivalence.)
    """
    masked = _model((3, 3), seed=11, modes=[[ZERO, RELU, RELU], [RELU] * 3])
    cut = _model((3, 3), seed=11)
    with torch.no_grad():
        cut.linears[1].weight[:, 0] = 0.0
    lo, hi = _box(13)
    masked_l2 = interval_ranges(masked, lo, hi)[1]
    cut_l2 = interval_ranges(cut, lo, hi)[1]
    torch.testing.assert_close(masked_l2.pre_lower, cut_l2.pre_lower)
    torch.testing.assert_close(masked_l2.pre_upper, cut_l2.pre_upper)


def test_certified_modes_flags_provably_stable_neurons() -> None:
    """Large biases against a tiny box certify IDENTITY / ZERO."""
    model = _model((3,), seed=17)
    with torch.no_grad():
        model.linears[0].bias.copy_(torch.tensor([100.0, -100.0, 0.0]).double())
    lo, hi = _box(19)
    for method in (interval_ranges, lp_ranges):
        modes = certified_modes(method(model, lo, hi))
        assert int(modes["relu0"][0]) == IDENT
        assert int(modes["relu0"][1]) == ZERO
        assert int(modes["relu0"][2]) == RELU


def test_certified_neurons_hold_on_sampled_inputs() -> None:
    """Certified-IDENTITY/ZERO neurons keep their sign on every sample."""
    model = _model((6, 4), seed=23)
    lo, hi = _box(29)
    ranges = lp_ranges(model, lo, hi)
    modes = certified_modes(ranges)
    samples = _sampled_pre_acts(model, lo, hi, n=128, seed=31)
    for layer, z in zip(ranges, samples, strict=True):
        mode = modes[layer.site]
        assert bool((z[:, mode == IDENT] > 0).all())
        assert bool((z[:, mode == ZERO] <= 0).all())


def test_data_box_covers_the_dataset() -> None:
    torch.manual_seed(37)
    x = torch.randn(32, INPUT_DIM)
    loader = DataLoader(
        TensorDataset(x, torch.zeros(32, dtype=torch.int64)), batch_size=8
    )
    lo, hi = data_box(loader)
    torch.testing.assert_close(lo, x.double().min(dim=0).values)
    torch.testing.assert_close(hi, x.double().max(dim=0).values)


def test_data_box_rejects_empty_loader() -> None:
    loader = DataLoader(
        TensorDataset(torch.zeros(0, 3), torch.zeros(0, dtype=torch.int64)),
        batch_size=4,
    )
    with pytest.raises(ValueError, match="non-empty"):
        data_box(loader)


def test_invalid_box_is_rejected() -> None:
    model = _model((3,), seed=41)
    lo = torch.ones(INPUT_DIM, dtype=torch.float64)
    with pytest.raises(ValueError, match="lower > upper"):
        interval_ranges(model, lo, lo - 1.0)
    with pytest.raises(ValueError, match="shapes differ"):
        interval_ranges(model, lo, torch.ones(INPUT_DIM + 1, dtype=torch.float64))
