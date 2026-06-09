"""Unit tests for regularization.loss (spec §4.4, §6 I3, §7, [R1])."""

import torch

from kill_nonlinearities.regularization.entropy import (
    batch_fraction_positive,
    binary_entropy,
)
from kill_nonlinearities.regularization.loss import sign_consistency_loss


def test_sign_consistency_loss_is_nonnegative_scalar() -> None:
    """The loss is a non-negative scalar (spec §7)."""
    torch.manual_seed(0)
    pre = [torch.randn(8, 4), torch.randn(8, 3)]
    out = sign_consistency_loss(pre, tau=1.0, eps=1e-6)
    assert out.ndim == 0
    assert out.item() >= 0.0


def test_sign_consistency_loss_matches_hand_computed_interior() -> None:
    """Two-site interior input matches the documented per-site mean-of-means (§7 [R-nit])."""
    tau = 1.0
    eps = 1e-6
    site_a = torch.tensor([[1.0, -1.0], [-2.0, 2.0]])
    site_b = torch.tensor([[0.5, -0.5, 1.5]])
    pre = [site_a, site_b]

    def site_entropy(z: torch.Tensor) -> torch.Tensor:
        p = batch_fraction_positive(z, tau).clamp(eps, 1.0 - eps)
        return binary_entropy(p).mean()

    expected = torch.stack([site_entropy(site_a), site_entropy(site_b)]).mean()
    out = sign_consistency_loss(pre, tau=tau, eps=eps)
    torch.testing.assert_close(out, expected)


def test_sign_consistency_loss_finite_gradient_under_saturation() -> None:
    """Saturating |z|/tau ~ 20 (float32) -> finite loss and finite grads ([R1], §7)."""
    tau = 0.1
    # |z| / tau == 20 -> sigmoid saturates to exactly 1.0 / 0.0 in float32.
    site_a = torch.tensor(
        [[2.0, -2.0], [2.0, -2.0], [2.0, -2.0]],
        dtype=torch.float32,
        requires_grad=True,
    )
    site_b = torch.tensor([[-2.0, 2.0, 2.0]], dtype=torch.float32, requires_grad=True)
    loss = sign_consistency_loss([site_a, site_b], tau=tau, eps=1e-6)
    assert torch.isfinite(loss)
    loss.backward()
    for site in (site_a, site_b):
        assert site.grad is not None
        assert torch.all(torch.isfinite(site.grad))


def test_sign_consistency_loss_saturated_gradient_is_zero() -> None:
    """Fully-saturated neurons get a finite ZERO gradient (clamp backward, [R1])."""
    tau = 0.1
    z = torch.full((4, 2), 2.0, dtype=torch.float32, requires_grad=True)
    sign_consistency_loss([z], tau=tau, eps=1e-6).backward()
    assert z.grad is not None
    torch.testing.assert_close(z.grad, torch.zeros_like(z))
