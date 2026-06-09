"""Entropy and fraction-positive helpers in nats (spec §4.4).

``binary_entropy`` uses ``torch.special.xlogy`` so the FORWARD value is exactly 0 at
``p ∈ {0, 1}``. Its BACKWARD is ``log((1-p)/p) → ±inf`` at the endpoints; that NaN
gradient is intentional honest math used only by no-grad analysis. The loss path
(``loss.py``) clamps ``p`` before calling this (spec §4.4 [R1]).
"""

import torch
from torch import Tensor


def binary_entropy(p: Tensor) -> Tensor:
    """Binary entropy in nats: ``-(xlogy(p, p) + xlogy(1-p, 1-p))`` (spec §4.4)."""
    return -(torch.special.xlogy(p, p) + torch.special.xlogy(1.0 - p, 1.0 - p))
