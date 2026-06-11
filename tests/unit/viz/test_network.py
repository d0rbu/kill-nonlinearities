"""Unit tests for kill_nonlinearities.viz.network (phase-3 figures)."""

from pathlib import Path

import pytest
import torch

from kill_nonlinearities.analysis.statistics import NeuronStats
from kill_nonlinearities.surgery.fold import FoldedMLP, fold_mlp, trim_folded
from kill_nonlinearities.viz import network


def _stats() -> list[NeuronStats]:
    return [
        NeuronStats(site="relu0", index=0, q=0.0, entropy=0.0, mean_pre=-1.0),
        NeuronStats(site="relu0", index=1, q=1.0, entropy=0.0, mean_pre=2.0),
        NeuronStats(site="relu0", index=2, q=0.5, entropy=0.6931, mean_pre=0.0),
        NeuronStats(site="relu1", index=0, q=0.001, entropy=0.0079, mean_pre=-3.0),
        NeuronStats(site="relu1", index=1, q=0.999, entropy=0.0079, mean_pre=3.0),
    ]


def test_categorize_buckets_every_unit_exactly_once() -> None:
    counts = network._categorize(_stats(), low_entropy=0.05)
    assert counts == {
        "exact dead": 1,
        "near dead": 1,
        "switching": 1,
        "near on": 1,
        "exact on": 1,
    }
    assert sum(counts.values()) == len(_stats())


def test_plot_mode_composition_writes_nonempty_file(tmp_path: Path) -> None:
    out = tmp_path / "composition.png"
    result = network.plot_mode_composition(_stats(), low_entropy=0.05, path=out)
    assert result == out
    assert out.stat().st_size > 0


def test_plot_spatial_q_map_accepts_flat_and_shaped_q(tmp_path: Path) -> None:
    torch.manual_seed(0)
    q = torch.rand(6 * 4 * 4)
    flat = network.plot_spatial_q_map(q, side=4, path=tmp_path / "flat.png")
    shaped = network.plot_spatial_q_map(
        q.reshape(6, 4, 4), side=4, path=tmp_path / "shaped.png"
    )
    assert flat.stat().st_size > 0
    assert shaped.stat().st_size > 0


def test_plot_spatial_q_map_caps_channels(tmp_path: Path) -> None:
    q = torch.rand(20, 3, 3)
    out = network.plot_spatial_q_map(
        q, side=3, path=tmp_path / "capped.png", max_channels=4
    )
    assert out.stat().st_size > 0


def test_plot_input_filters_grayscale_and_rgb(tmp_path: Path) -> None:
    torch.manual_seed(1)
    gray = network.plot_input_filters(
        torch.randn(5, 16), image_shape=(4, 4), path=tmp_path / "gray.png"
    )
    rgb = network.plot_input_filters(
        torch.randn(3, 48), image_shape=(3, 4, 4), path=tmp_path / "rgb.png"
    )
    assert gray.stat().st_size > 0
    assert rgb.stat().st_size > 0


def test_plot_input_filters_handles_constant_zero_filter(tmp_path: Path) -> None:
    out = network.plot_input_filters(
        torch.zeros(2, 9), image_shape=(3, 3), path=tmp_path / "zero.png"
    )
    assert out.stat().st_size > 0


def test_plot_input_filters_rejects_empty_weight(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no rows"):
        network.plot_input_filters(
            torch.zeros(0, 9), image_shape=(3, 3), path=tmp_path / "none.png"
        )


def test_plot_class_q_matrix_writes_nonempty_file(tmp_path: Path) -> None:
    torch.manual_seed(2)
    q = torch.rand(10, 24)
    out = network.plot_class_q_matrix(q, tmp_path / "class_q.png")
    assert out.stat().st_size > 0


def test_plot_class_q_matrix_accepts_class_names(tmp_path: Path) -> None:
    q = torch.rand(3, 5)
    out = network.plot_class_q_matrix(
        q, tmp_path / "named.png", class_names=("cat", "dog", "frog")
    )
    assert out.stat().st_size > 0


def test_plot_class_q_matrix_rejects_non_2d_input(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="classes, neurons"):
        network.plot_class_q_matrix(torch.rand(8), tmp_path / "bad.png")


def test_plot_spatial_q_map_panel_labels_override_titles(tmp_path: Path) -> None:
    q = torch.rand(3, 4, 4)
    out = network.plot_spatial_q_map(
        q,
        side=4,
        path=tmp_path / "labeled.png",
        panel_labels=("plane", "car", "bird"),
    )
    assert out.stat().st_size > 0


def _toy_tree():
    from kill_nonlinearities.analysis.decompile import Branch, Leaf, Truncated

    leaf = Leaf(weight=torch.zeros(2, 3), bias=torch.zeros(2))
    inner = Branch(
        site="relu0",
        index=1,
        weight=torch.ones(3),
        bias=0.5,
        low=Truncated(),
        high=Leaf(weight=torch.ones(2, 3), bias=torch.ones(2)),
    )
    return Branch(
        site="relu0", index=0, weight=torch.ones(3), bias=-0.5, low=leaf, high=inner
    )


def test_plot_decision_tree_writes_nonempty_file(tmp_path: Path) -> None:
    out = network.plot_decision_tree(_toy_tree(), tmp_path / "tree.png")
    assert out.stat().st_size > 0


def test_plot_decision_tree_accepts_leaf_labeler_and_single_leaf(
    tmp_path: Path,
) -> None:
    from kill_nonlinearities.analysis.decompile import Leaf

    leaf = Leaf(weight=torch.zeros(2, 3), bias=torch.zeros(2))
    out = network.plot_decision_tree(
        leaf, tmp_path / "leaf.png", leaf_label=lambda _: "class 7"
    )
    assert out.stat().st_size > 0


def test_plot_decision_tree_refuses_huge_trees(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="terminal nodes"):
        network.plot_decision_tree(_toy_tree(), tmp_path / "huge.png", max_terminals=1)


def _folded(hidden: tuple[int, ...] = (5, 3)) -> FoldedMLP:
    from kill_nonlinearities.config import ModelConfig
    from kill_nonlinearities.models.mlp import ReLUMLP

    torch.manual_seed(4)
    model = ReLUMLP(ModelConfig(input_dim=9, hidden_dims=hidden, output_dim=3))
    return trim_folded(fold_mlp(model))


def test_plot_folded_dag_writes_nonempty_file(tmp_path: Path) -> None:
    out = network.plot_folded_dag(_folded(), tmp_path / "dag.png")
    assert out.stat().st_size > 0


def test_plot_folded_dag_with_image_glyphs_and_class_names(tmp_path: Path) -> None:
    out = network.plot_folded_dag(
        _folded(),
        tmp_path / "dag_glyphs.png",
        image_shape=(3, 3),
        class_names=("a", "b", "c"),
    )
    assert out.stat().st_size > 0


def test_plot_folded_dag_refuses_wide_networks(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="surviving units"):
        network.plot_folded_dag(_folded(), tmp_path / "wide.png", max_units=2)
