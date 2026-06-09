"""Entropy and fraction-positive helpers in nats (spec §4.4).

``binary_entropy`` uses ``torch.special.xlogy`` so the FORWARD value is exactly 0 at
``p ∈ {0, 1}``. Its BACKWARD is ``log((1-p)/p) → ±inf`` at the endpoints; that NaN
gradient is intentional honest math used only by no-grad analysis. The loss path
(``loss.py``) clamps ``p`` before calling this (spec §4.4 [R1]).
"""

import torch
from torch import Tensor

from kill_nonlinearities.regularization.surrogate import soft_sign


def binary_entropy(p: Tensor) -> Tensor:
    """Binary entropy in nats: ``-(xlogy(p, p) + xlogy(1-p, 1-p))`` (spec §4.4)."""
    return -(torch.special.xlogy(p, p) + torch.special.xlogy(1.0 - p, 1.0 - p))


def batch_fraction_positive(z: Tensor, tau: float) -> Tensor:
    """Soft fraction-positive ``p_i = soft_sign(z, tau).mean(0)`` -> ``[N]``.

    Requires ``z`` to have at least one row (B >= 1). ``tau`` is validated by
    ``soft_sign`` (spec §4.4).
    """
    return soft_sign(z, tau).mean(0)


def hard_fraction_positive(z: Tensor) -> Tensor:
    """Hard fraction-positive ``q_i = (z > 0).mean(0)`` -> ``[N]``.

    Uses the strict ``>`` predicate, so ``z == 0`` counts as negative (spec §4.4 [R6]).
    """
    return (z > 0.0).to(z.dtype).mean(0)
