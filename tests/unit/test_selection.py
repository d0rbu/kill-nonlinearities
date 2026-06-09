"""Unit tests for analysis.selection ranking/selection (spec §4.10, §7)."""

import torch

from kill_nonlinearities.analysis.selection import (
    assign_modes,
    rank_by_entropy,
    rank_random,
    select_topk,
)
from kill_nonlinearities.analysis.statistics import NeuronStats
from kill_nonlinearities.models.activations import ActivationMode


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


def test_select_topk_boundaries_global_across_sites() -> None:
    ranked = rank_by_entropy(_stats())  # ordered: relu0/1, relu1/0, relu0/0
    assert select_topk(ranked, 0) == []
    assert select_topk(ranked, len(ranked)) == ranked
    top1 = select_topk(ranked, 1)
    assert len(top1) == 1
    assert (top1[0].site, top1[0].index) == ("relu0", 1)
    # k spans sites globally: top-2 pulls from two different sites.
    top2 = select_topk(ranked, 2)
    assert {(s.site, s.index) for s in top2} == {("relu0", 1), ("relu1", 0)}


def test_assign_modes_full_width_and_rules() -> None:
    # Two sites: relu0 width 2, relu1 width 1.
    selection = [
        NeuronStats(site="relu0", index=0, q=0.0, entropy=0.0, mean_pre=-1.0),
        NeuronStats(site="relu0", index=1, q=1.0, entropy=0.0, mean_pre=3.0),
        NeuronStats(site="relu1", index=0, q=0.5, entropy=0.69, mean_pre=0.0),
    ]
    # True site widths: relu0 has 2 neurons, relu1 has 1.
    modes = assign_modes(
        selection, tie_break="identity", widths={"relu0": 2, "relu1": 1}
    )

    assert set(modes) == {"relu0", "relu1"}
    assert modes["relu0"].dtype == torch.int64
    assert modes["relu0"].shape == (2,)
    assert modes["relu1"].shape == (1,)
    # q<0.5 -> ZERO; q>0.5 -> IDENTITY; q==0.5 -> tie_break (identity here).
    assert modes["relu0"][0].item() == int(ActivationMode.ZERO)
    assert modes["relu0"][1].item() == int(ActivationMode.IDENTITY)
    assert modes["relu1"][0].item() == int(ActivationMode.IDENTITY)


def test_assign_modes_unselected_stay_relu() -> None:
    # relu0 has true width 3 but only index 2 is selected -> 0 and 1 stay RELU.
    selection = [
        NeuronStats(site="relu0", index=2, q=0.0, entropy=0.0, mean_pre=-1.0),
    ]
    modes = assign_modes(selection, tie_break="zero", widths={"relu0": 3})
    assert modes["relu0"].shape == (3,)
    assert modes["relu0"][0].item() == int(ActivationMode.RELU)
    assert modes["relu0"][1].item() == int(ActivationMode.RELU)
    assert modes["relu0"][2].item() == int(ActivationMode.ZERO)


def test_assign_modes_tie_break_zero() -> None:
    selection = [
        NeuronStats(site="relu0", index=0, q=0.5, entropy=0.69, mean_pre=0.0),
    ]
    modes = assign_modes(selection, tie_break="zero", widths={"relu0": 1})
    assert modes["relu0"][0].item() == int(ActivationMode.ZERO)
