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
from kill_nonlinearities.regularization.surrogate import soft_sign


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
    if len(pre_activations) == 0:
        raise ValueError("sign_consistency_loss requires at least one site")
    site_entropies: list[Tensor] = []
    for z in pre_activations:
        p = batch_fraction_positive(z, tau).clamp(eps, 1.0 - eps)
        site_entropies.append(binary_entropy(p).mean())
    return torch.stack(site_entropies).mean()


def grouped_sign_consistency_loss(
    pre_activations: Sequence[Tensor],
    group_sizes: Sequence[int],
    tau: float,
    eps: float,
) -> Tensor:
    """Channel-pooled variant: per-site mean entropy of per-GROUP fraction-positive.

    Each site's ``[B, N]`` pre-activations are split into ``N / g`` contiguous
    groups of ``g`` neurons (for ``ReLUCNN`` conv sites one group is a channel:
    the flatten convention ``i = (c*H + h)*W + w`` keeps a channel's positions
    contiguous, and ``g = H*W``); the soft fraction-positive pools over batch
    AND group (``B * g`` samples) before the entropy. With ``g = 1`` at every
    site this equals ``sign_consistency_loss``. Same clamp-before-entropy
    gradient-safety rule ([R1]).

    Note the deliberate semantic difference from the per-neuron loss (conv spec
    2026-06-10): a group whose members are individually sign-consistent in
    *mixed* directions has interior pooled ``p`` and is still penalized — this
    variant incentivizes whole groups to commit one way.

    Args:
        pre_activations: One ``[B, N_l]`` pre-activation tensor per site.
        group_sizes: Neurons per group at each site; must divide that site's
            ``N_l``. Length must match ``pre_activations``.
        tau: Surrogate temperature (validated by ``soft_sign``; must be > 0).
        eps: Loss-path clamp bound ([R1]).

    Returns:
        A scalar loss tensor (mean over sites of each site's mean group entropy).
    """
    if len(pre_activations) == 0:
        raise ValueError("grouped_sign_consistency_loss requires at least one site")
    if len(group_sizes) != len(pre_activations):
        raise ValueError(
            f"group_sizes has {len(group_sizes)} entries for "
            f"{len(pre_activations)} sites"
        )
    site_entropies: list[Tensor] = []
    for z, g in zip(pre_activations, group_sizes, strict=True):
        n = int(z.shape[1])
        if g < 1 or n % g != 0:
            raise ValueError(f"group size {g} must divide the site width {n}")
        p = soft_sign(z, tau).reshape(z.shape[0], n // g, g).mean(dim=(0, 2))
        p = p.clamp(eps, 1.0 - eps)
        site_entropies.append(binary_entropy(p).mean())
    return torch.stack(site_entropies).mean()
