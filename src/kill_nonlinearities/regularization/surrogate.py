"""Differentiable sign surrogate (spec §4.4)."""

import torch
from torch import Tensor


def soft_sign(z: Tensor, tau: float) -> Tensor:
    """Smooth surrogate for the sign indicator: ``sigmoid(z / tau)``.

    Args:
        z: Pre-activations of any shape.
        tau: Temperature; must be strictly positive (spec §4.4 [R16]).

    Returns:
        ``sigmoid(z / tau)``, elementwise in ``(0, 1)``.

    Raises:
        ValueError: If ``tau <= 0``.
    """
    if tau <= 0.0:
        raise ValueError(f"tau must be > 0, got {tau}")
    return torch.sigmoid(z / tau)
