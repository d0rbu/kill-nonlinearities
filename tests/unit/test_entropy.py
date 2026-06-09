"""Unit tests for regularization.entropy (spec §4.4, §6 I3, §7)."""

import math

import torch

from kill_nonlinearities.regularization.entropy import binary_entropy


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
