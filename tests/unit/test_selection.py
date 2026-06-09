"""Unit tests for analysis.selection ranking/selection (spec §4.10, §7)."""

from kill_nonlinearities.analysis.selection import (
    rank_by_entropy,
    rank_random,
)
from kill_nonlinearities.analysis.statistics import NeuronStats


def _stats() -> list[NeuronStats]:
    return [
        NeuronStats(site="relu1", index=0, q=0.0, entropy=0.5, mean_pre=1.0),
        NeuronStats(site="relu0", index=1, q=1.0, entropy=0.1, mean_pre=2.0),
        NeuronStats(site="relu0", index=0, q=0.5, entropy=0.9, mean_pre=-1.0),
    ]


def test_rank_by_entropy_ascending() -> None:
    ranked = rank_by_entropy(_stats())
    assert [s.entropy for s in ranked] == [0.1, 0.5, 0.9]


def test_rank_by_entropy_real_tie_broken_by_site_then_index() -> None:
    # Byte-identical entropy across two sites and within a site.
    tied = [
        NeuronStats(site="relu1", index=0, q=0.0, entropy=0.3, mean_pre=0.0),
        NeuronStats(site="relu0", index=1, q=0.0, entropy=0.3, mean_pre=0.0),
        NeuronStats(site="relu0", index=0, q=0.0, entropy=0.3, mean_pre=0.0),
    ]
    ranked = rank_by_entropy(tied)
    assert [(s.site, s.index) for s in ranked] == [
        ("relu0", 0),
        ("relu0", 1),
        ("relu1", 0),
    ]


def test_rank_random_is_deterministic_under_seed() -> None:
    stats = _stats()
    a = rank_random(stats, seed=7)
    b = rank_random(stats, seed=7)
    assert a == b
    # A permutation of the same neurons (no loss/dup).
    assert sorted((s.site, s.index) for s in a) == sorted(
        (s.site, s.index) for s in stats
    )
