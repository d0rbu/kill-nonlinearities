"""Unit tests for regularization.loss (spec §4.4, §6 I3, §7, [R1])."""

import pytest
import torch

from kill_nonlinearities.regularization.entropy import (
    batch_fraction_positive,
    binary_entropy,
)
from kill_nonlinearities.regularization.loss import (
    grouped_sign_consistency_loss,
    sign_consistency_loss,
)
from kill_nonlinearities.regularization.surrogate import soft_sign


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


def test_sign_consistency_loss_single_site_is_that_sites_mean_entropy() -> None:
    """Single-site loss == that site's mean clamped per-neuron entropy (spec §7)."""
    tau = 1.0
    eps = 1e-6
    z = torch.tensor([[1.0, -1.0, 0.5], [-2.0, 2.0, -0.5]])
    p = batch_fraction_positive(z, tau).clamp(eps, 1.0 - eps)
    expected = binary_entropy(p).mean()
    out = sign_consistency_loss([z], tau=tau, eps=eps)
    torch.testing.assert_close(out, expected)


def test_sign_consistency_loss_gradient_flows_to_all_pre_activations() -> None:
    """loss.backward() produces finite, non-zero grads on interior pre-acts (spec §7)."""
    tau = 1.0
    site_a = torch.tensor(
        [[0.5, -0.5], [-0.3, 0.3]], dtype=torch.float32, requires_grad=True
    )
    site_b = torch.tensor([[0.2, -0.4, 0.1]], dtype=torch.float32, requires_grad=True)
    sign_consistency_loss([site_a, site_b], tau=tau, eps=1e-6).backward()
    for site in (site_a, site_b):
        assert site.grad is not None
        assert torch.all(torch.isfinite(site.grad))
        assert torch.any(site.grad != 0.0)


def test_sign_consistency_loss_invariant_to_batch_permutation() -> None:
    """Permuting rows (batch) leaves the scalar loss unchanged (assert_close, [R13])."""
    torch.manual_seed(1)
    z = torch.randn(10, 4)
    perm = torch.randperm(10)
    base = sign_consistency_loss([z], tau=1.0, eps=1e-6)
    permuted = sign_consistency_loss([z[perm]], tau=1.0, eps=1e-6)
    torch.testing.assert_close(base, permuted)


def test_sign_consistency_loss_invariant_to_neuron_permutation() -> None:
    """Permuting columns (neurons) leaves the scalar loss unchanged (assert_close, [R13])."""
    torch.manual_seed(2)
    z = torch.randn(10, 4)
    perm = torch.randperm(4)
    base = sign_consistency_loss([z], tau=1.0, eps=1e-6)
    permuted = sign_consistency_loss([z[:, perm]], tau=1.0, eps=1e-6)
    torch.testing.assert_close(base, permuted)


def test_per_neuron_entropy_vector_invariant_under_inverse_neuron_perm() -> None:
    """The per-neuron entropy vector is invariant under the inverse perm (assert_close, [R13])."""
    torch.manual_seed(3)
    tau = 1.0
    eps = 1e-6
    z = torch.randn(10, 5)
    perm = torch.randperm(5)
    inv = torch.argsort(perm)

    def per_neuron_entropy(zz: torch.Tensor) -> torch.Tensor:
        p = batch_fraction_positive(zz, tau).clamp(eps, 1.0 - eps)
        return binary_entropy(p)

    base_vec = per_neuron_entropy(z)
    permuted_vec = per_neuron_entropy(z[:, perm])
    # ``mean(0)`` over a column-permuted (non-contiguous) tensor is NOT bit-stable:
    # the vectorized reduction accumulates in a different order, yielding ~1-ULP
    # per-column differences. Per §0 (and this milestone's stated convention),
    # permutation invariance is asserted via ``assert_close``, not ``torch.equal``.
    torch.testing.assert_close(permuted_vec[inv], base_vec)


def test_sign_consistency_loss_rejects_empty_sites() -> None:
    """An empty pre_activations sequence is a usage error (spec §7 [R22])."""
    with pytest.raises(ValueError, match="at least one"):
        sign_consistency_loss([], tau=1.0, eps=1e-6)


def test_grouped_loss_with_unit_groups_equals_per_neuron_loss() -> None:
    """g=1 at every site reduces the grouped loss to sign_consistency_loss."""
    torch.manual_seed(4)
    pre = [torch.randn(8, 4), torch.randn(8, 3)]
    grouped = grouped_sign_consistency_loss(pre, [1, 1], tau=1.0, eps=1e-6)
    base = sign_consistency_loss(pre, tau=1.0, eps=1e-6)
    torch.testing.assert_close(grouped, base)


def test_grouped_loss_matches_hand_computed_pooling() -> None:
    """One site, N=4, g=2: p pools sigmoid over batch AND the 2-neuron group."""
    tau = 1.0
    eps = 1e-6
    z = torch.tensor([[1.0, -1.0, 0.5, 2.0], [-2.0, 2.0, -0.5, 1.0]])
    s = soft_sign(z, tau)
    p = torch.stack(
        [s[:, 0:2].mean(), s[:, 2:4].mean()]  # one pooled p per group
    ).clamp(eps, 1.0 - eps)
    expected = binary_entropy(p).mean()
    out = grouped_sign_consistency_loss([z], [2], tau=tau, eps=eps)
    torch.testing.assert_close(out, expected)


def test_grouped_loss_penalizes_mixed_direction_consistent_group() -> None:
    """The conv-spec counterexample: per-neuron loss ~0, grouped loss ~ln 2.

    Two neurons in one group, each perfectly sign-consistent across the batch
    but in OPPOSITE directions: the per-neuron loss saturates to ~0 while the
    channel-pooled loss reads p=0.5 and pays maximal entropy — the deliberate
    semantic difference between the two granularities.
    """
    tau = 0.1
    eps = 1e-6
    # sigmoid(+-2 / 0.1) saturates to exactly 1.0 / 0.0 in float32.
    z = torch.tensor([[2.0, -2.0], [2.0, -2.0], [2.0, -2.0], [2.0, -2.0]])
    per_neuron = sign_consistency_loss([z], tau=tau, eps=eps)
    grouped = grouped_sign_consistency_loss([z], [2], tau=tau, eps=eps)
    assert per_neuron.item() < 1e-4
    torch.testing.assert_close(grouped, torch.tensor(0.5).log().neg())  # ln 2


def test_grouped_loss_finite_gradient_under_saturation() -> None:
    """Saturated groups get finite (zero) gradients via the clamp ([R1])."""
    z = torch.full((4, 4), 2.0, dtype=torch.float32, requires_grad=True)
    loss = grouped_sign_consistency_loss([z], [2], tau=0.1, eps=1e-6)
    assert torch.isfinite(loss)
    loss.backward()
    assert z.grad is not None
    torch.testing.assert_close(z.grad, torch.zeros_like(z))


def test_grouped_loss_rejects_empty_sites() -> None:
    with pytest.raises(ValueError, match="at least one"):
        grouped_sign_consistency_loss([], [], tau=1.0, eps=1e-6)


def test_grouped_loss_rejects_mismatched_group_sizes_length() -> None:
    with pytest.raises(ValueError, match="entries for"):
        grouped_sign_consistency_loss([torch.randn(4, 4)], [2, 2], tau=1.0, eps=1e-6)


def test_grouped_loss_rejects_non_dividing_group_size() -> None:
    with pytest.raises(ValueError, match="must divide"):
        grouped_sign_consistency_loss([torch.randn(4, 4)], [3], tau=1.0, eps=1e-6)
