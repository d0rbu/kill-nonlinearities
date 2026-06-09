"""Surgery selection: ranking, k-grid, top-k, and mode assignment (spec §4.10)."""

import torch

from kill_nonlinearities.analysis.statistics import NeuronStats


def make_k_grid(total: int, num_k: int) -> list[int]:
    """Target ``num_k`` evenly-spaced k-values in ``[0, total]`` (spec §4.10).

    Always includes ``0`` and ``total``. The realized length equals the number
    of UNIQUE rounded points, which may be fewer than ``num_k`` on tiny models.
    """
    return sorted({round(i * total / (num_k - 1)) for i in range(num_k)})


def rank_by_entropy(stats: list[NeuronStats]) -> list[NeuronStats]:
    """Ascending sign-entropy, tie-broken deterministically (spec §4.10)."""
    return sorted(stats, key=lambda s: (s.entropy, s.site, s.index))


def rank_random(stats: list[NeuronStats], seed: int) -> list[NeuronStats]:
    """Seeded random permutation of ``stats`` (control baseline, spec §4.10)."""
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(stats), generator=generator).tolist()
    return [stats[i] for i in order]
