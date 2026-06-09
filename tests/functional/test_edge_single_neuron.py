"""N=1 site edge case through loss, stats, and ranking (spec §7 [R22])."""

import torch

from kill_nonlinearities.analysis.selection import rank_by_entropy
from kill_nonlinearities.analysis.statistics import neuron_stats
from kill_nonlinearities.regularization.loss import sign_consistency_loss


def test_single_neuron_site_loss_is_finite_scalar() -> None:
    z = torch.tensor([[2.0], [-3.0], [1.0]])  # batch=3, N=1
    loss = sign_consistency_loss([z], tau=1.0, eps=1e-6)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_single_neuron_site_stats_and_ranking() -> None:
    # one site "relu0" with a single neuron, all-positive over the batch -> q=1, entropy 0
    pre_by_site = {"relu0": torch.tensor([[1.0], [2.0], [0.5]])}
    stats = neuron_stats(pre_by_site)
    assert len(stats) == 1
    assert stats[0].site == "relu0"
    assert stats[0].index == 0
    assert stats[0].q == 1.0
    assert stats[0].entropy == 0.0
    ranked = rank_by_entropy(stats)
    assert [s.index for s in ranked] == [0]
