"""Unit tests for regularization.entropy (spec §4.4, §6 I3, §7)."""

import math

import torch

from kill_nonlinearities.regularization.entropy import (
    batch_fraction_positive,
    binary_entropy,
    hard_fraction_positive,
)


def test_binary_entropy_zero_at_endpoints_exact() -> None:
    """H(0) == H(1) == 0 exactly (xlogy forward value, spec §6 I3)."""
    p = torch.tensor([0.0, 1.0])
    out = binary_entropy(p)
    assert torch.equal(out, torch.zeros(2))


def test_binary_entropy_half_is_ln2() -> None:
    """H(0.5) == ln(2) (nats, spec §6 I3)."""
    out = binary_entropy(torch.tensor(0.5))
    torch.testing.assert_close(out, torch.tensor(math.log(2.0)))


def test_binary_entropy_is_symmetric() -> None:
    """H(p) == H(1 - p) (spec §6 I3, §7)."""
    p = torch.tensor([0.1, 0.25, 0.4, 0.7, 0.9])
    torch.testing.assert_close(binary_entropy(p), binary_entropy(1.0 - p))


def test_binary_entropy_closed_form_at_quarter() -> None:
    """H(0.25) matches the closed form -(0.25 ln0.25 + 0.75 ln0.75) (spec §7)."""
    expected = -(0.25 * math.log(0.25) + 0.75 * math.log(0.75))
    out = binary_entropy(torch.tensor(0.25))
    torch.testing.assert_close(out, torch.tensor(expected))


def test_binary_entropy_mixes_endpoints_and_interior_finite() -> None:
    """A tensor mixing endpoints and interior is finite, exact at ends (spec §7 [R22])."""
    p = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
    out = binary_entropy(p)
    assert torch.all(torch.isfinite(out))
    assert out[0].item() == 0.0
    assert out[-1].item() == 0.0
    torch.testing.assert_close(out[2], torch.tensor(math.log(2.0)))


def test_binary_entropy_gradient_closed_form_interior() -> None:
    """dH/dp == log((1-p)/p) on the open interval (spec §6 I3 [R12])."""
    p = torch.linspace(0.05, 0.95, 19, dtype=torch.float64, requires_grad=True)
    binary_entropy(p).sum().backward()
    expected = torch.log((1.0 - p) / p)
    assert p.grad is not None
    torch.testing.assert_close(p.grad, expected)


def test_binary_entropy_gradcheck_open_interval_float64() -> None:
    """gradcheck passes on p in [0.05, 0.95], float64 (spec §6 I3 [R12])."""
    p = torch.linspace(0.05, 0.95, 19, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(binary_entropy, (p,))


def test_batch_fraction_positive_shape_is_num_neurons() -> None:
    """batch_fraction_positive([B, N]) -> [N] (spec §4.4)."""
    z = torch.randn(8, 5)
    out = batch_fraction_positive(z, 1.0)
    assert out.shape == (5,)


def test_batch_fraction_positive_equals_mean_soft_sign() -> None:
    """p_i == soft_sign(z, tau).mean(0) (spec §4.4)."""
    z = torch.tensor([[1.0, -1.0], [2.0, -2.0], [0.5, 0.5]])
    expected = torch.sigmoid(z / 0.5).mean(0)
    torch.testing.assert_close(batch_fraction_positive(z, 0.5), expected)


def test_hard_fraction_positive_shape_and_values() -> None:
    """q_i == (z > 0).float().mean(0); strict '>' at 0 counts negative (spec §4.4 [R6])."""
    z = torch.tensor([[1.0, 0.0, -1.0], [2.0, -3.0, 4.0]])
    out = hard_fraction_positive(z)
    assert out.shape == (3,)
    # column 0: both > 0 -> 1.0; column 1: 0 and -3 -> 0/2; column 2: -1 and 4 -> 1/2
    torch.testing.assert_close(out, torch.tensor([1.0, 0.0, 0.5]))


def test_hard_fraction_positive_in_unit_interval() -> None:
    """q_i lies in [0, 1] (spec §7)."""
    z = torch.randn(16, 7)
    out = hard_fraction_positive(z)
    assert torch.all(out >= 0.0)
    assert torch.all(out <= 1.0)


def test_hard_fraction_positive_zero_counts_as_negative() -> None:
    """A column of all-zeros yields q == 0 (strict '>', spec §4.4 [R6])."""
    z = torch.zeros(4, 3)
    torch.testing.assert_close(hard_fraction_positive(z), torch.zeros(3))
