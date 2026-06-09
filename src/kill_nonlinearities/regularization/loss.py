"""Sign-consistency regularization loss (spec §4.4).

The loss clamps the SOFT fraction-positive ``p`` into ``[eps, 1-eps]`` BEFORE
``binary_entropy`` (BLOCKER fix [R1]): at low ``tau`` a sign-consistent neuron
saturates ``p`` to exactly 0.0/1.0 in float32, whose entropy backward is
``log((1-p)/p) -> ±inf`` (NaN). Clamp's backward is 0 outside the range, so
fully-saturated neurons get a finite (zero) gradient. ``binary_entropy`` itself
stays unclamped and is used by analysis on the hard ``q`` under no-grad.

Per-site equal weighting (mean over sites of each site's per-neuron mean entropy)
is intentional and matches ``1/|L| Σ_l 1/N_l Σ_i`` (spec §4.4 [R28]).
"""

from collections.abc import Sequence

import torch
from torch import Tensor

from kill_nonlinearities.regularization.entropy import (
    batch_fraction_positive,
    binary_entropy,
)


def sign_consistency_loss(
    pre_activations: Sequence[Tensor], tau: float, eps: float
) -> Tensor:
    """Mean over sites of per-neuron clamped sign-entropy (spec §4.4).

    Args:
        pre_activations: One ``[B, N_l]`` pre-activation tensor per site.
        tau: Surrogate temperature (validated by ``soft_sign``; must be > 0).
        eps: Loss-path clamp bound; ``p`` is clamped to ``[eps, 1-eps]`` before
            entropy for gradient safety ([R1]).

    Returns:
        A scalar loss tensor (mean over sites of each site's mean entropy).
    """
    site_entropies: list[Tensor] = []
    for z in pre_activations:
        p = batch_fraction_positive(z, tau).clamp(eps, 1.0 - eps)
        site_entropies.append(binary_entropy(p).mean())
    return torch.stack(site_entropies).mean()
