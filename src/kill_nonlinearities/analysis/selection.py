"""Surgery selection: ranking, k-grid, top-k, and mode assignment (spec §4.10)."""

import torch
from torch import Tensor

from kill_nonlinearities.analysis.statistics import NeuronStats
from kill_nonlinearities.models.activations import ActivationMode


def make_k_grid(total: int, num_k: int) -> list[int]:
    """Target ``num_k`` evenly-spaced k-values in ``[0, total]`` (spec §4.10).

    Always includes ``0`` and ``total``. The realized length equals the number
    of UNIQUE rounded points, which may be fewer than ``num_k`` on tiny models.
    Requires ``num_k >= 2`` (the spacing divides by ``num_k - 1``).
    """
    if num_k < 2:
        raise ValueError(f"num_k must be >= 2, got {num_k}")
    return sorted({round(i * total / (num_k - 1)) for i in range(num_k)})


def rank_by_entropy(stats: list[NeuronStats]) -> list[NeuronStats]:
    """Ascending sign-entropy, tie-broken deterministically (spec §4.10)."""
    return sorted(stats, key=lambda s: (s.entropy, s.site, s.index))


def rank_random(stats: list[NeuronStats], seed: int) -> list[NeuronStats]:
    """Seeded random permutation of ``stats`` (control baseline, spec §4.10)."""
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(stats), generator=generator).tolist()
    return [stats[i] for i in order]


def select_topk(ranked: list[NeuronStats], k: int) -> list[NeuronStats]:
    """First ``k`` neurons of a GLOBAL ranking across all sites (spec §4.10)."""
    return ranked[:k]


def assign_modes(
    selection: list[NeuronStats],
    tie_break: str,
    widths: dict[str, int],
) -> dict[str, Tensor]:
    """Full-width ``[N_site]`` int64 mode tensors per site (spec §4.10, [R14][R17]).

    Initialized to ``RELU``; selected neurons become ``ZERO`` (q<0.5),
    ``IDENTITY`` (q>0.5), or ``tie_break`` (q==0.5). Unselected neurons stay
    ``RELU``. ``widths`` gives each site's true neuron count so the full-width
    tensors are correct regardless of which neurons are selected.
    """
    tie_mode = {
        "zero": ActivationMode.ZERO,
        "identity": ActivationMode.IDENTITY,
    }[tie_break]

    modes = {
        site: torch.full((width,), int(ActivationMode.RELU), dtype=torch.int64)
        for site, width in widths.items()
    }
    for s in selection:
        if s.q < 0.5:
            mode = ActivationMode.ZERO
        elif s.q > 0.5:
            mode = ActivationMode.IDENTITY
        else:
            mode = tie_mode
        modes[s.site][s.index] = int(mode)
    return modes
