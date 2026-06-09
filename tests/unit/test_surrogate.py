"""Unit tests for regularization.surrogate (spec §4.4, §7)."""

import importlib

import pytest
import torch

from kill_nonlinearities.regularization.surrogate import soft_sign


def test_regularization_package_importable() -> None:
    """The regularization subpackage imports cleanly."""
    module = importlib.import_module("kill_nonlinearities.regularization")
    assert module is not None


def test_soft_sign_equals_sigmoid_of_z_over_tau() -> None:
    """soft_sign(z, tau) == sigmoid(z / tau) (spec §4.4)."""
    z = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0])
    tau = 0.5
    torch.testing.assert_close(soft_sign(z, tau), torch.sigmoid(z / tau))


def test_soft_sign_at_zero_is_half() -> None:
    """z == 0 maps to exactly 0.5 (spec §4.4, §6 I3)."""
    z = torch.zeros(4)
    torch.testing.assert_close(soft_sign(z, 1.0), torch.full((4,), 0.5))


def test_soft_sign_in_open_unit_interval() -> None:
    """Output lies strictly in (0, 1) for finite inputs (spec §7)."""
    z = torch.linspace(-10.0, 10.0, 21)
    out = soft_sign(z, 1.0)
    assert torch.all(out > 0.0)
    assert torch.all(out < 1.0)


def test_soft_sign_monotone_increasing_in_z() -> None:
    """soft_sign is monotone increasing in z (spec §7)."""
    z = torch.linspace(-10.0, 10.0, 51)
    out = soft_sign(z, 1.0)
    assert torch.all(out[1:] - out[:-1] > 0.0)


def test_soft_sign_saturates_for_large_magnitude() -> None:
    """Large |z|/tau saturates toward 0 and 1 (spec §7)."""
    assert soft_sign(torch.tensor(50.0), 1.0).item() > 1.0 - 1e-6
    assert soft_sign(torch.tensor(-50.0), 1.0).item() < 1e-6


def test_soft_sign_raises_for_non_positive_tau() -> None:
    """tau <= 0 raises ValueError (spec §4.4 [R16])."""
    z = torch.zeros(3)
    with pytest.raises(ValueError, match="tau"):
        soft_sign(z, 0.0)
    with pytest.raises(ValueError, match="tau"):
        soft_sign(z, -1.0)


def test_soft_sign_gradcheck_float64_interior() -> None:
    """gradcheck passes on interior z in float64 (spec §6 I3, §7)."""
    z = torch.linspace(-3.0, 3.0, 7, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda zz: soft_sign(zz, 0.7), (z,))
