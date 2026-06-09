"""Unit tests for kill_nonlinearities.viz.plots (Agg headless figures)."""

from pathlib import Path

import matplotlib

from kill_nonlinearities.analysis.statistics import NeuronStats
from kill_nonlinearities.training.trainer import StepMetrics
from kill_nonlinearities.viz import plots


def test_importing_plots_selects_agg_backend() -> None:
    """Importing viz.plots forces the non-interactive Agg backend (spec §4.12)."""
    import kill_nonlinearities.viz.plots  # noqa: F401

    assert matplotlib.get_backend().lower() == "agg"


def _toy_history() -> list[StepMetrics]:
    return [
        StepMetrics(
            step=i,
            epoch=0,
            task_loss=1.0 - 0.1 * i,
            reg_loss=0.5 - 0.05 * i,
            total_loss=1.5 - 0.15 * i,
            tau=1.0 - 0.1 * i,
        )
        for i in range(3)
    ]


def test_plot_loss_curves_writes_nonempty_file(tmp_path: Path) -> None:
    """plot_loss_curves writes a non-empty figure file and returns its Path (spec §5)."""
    out = tmp_path / "loss_curves.png"
    result = plots.plot_loss_curves(_toy_history(), out)

    assert result == out
    assert out.exists()
    assert out.stat().st_size > 0


def _toy_stats() -> list[NeuronStats]:
    return [
        NeuronStats(site="relu0", index=0, q=0.0, entropy=0.0, mean_pre=-1.0),
        NeuronStats(site="relu0", index=1, q=1.0, entropy=0.0, mean_pre=2.0),
        NeuronStats(site="relu1", index=0, q=0.5, entropy=0.6931, mean_pre=0.1),
        NeuronStats(site="relu1", index=1, q=0.3, entropy=0.6109, mean_pre=-0.2),
    ]


def test_plot_mean_pre_dist_writes_nonempty_file(tmp_path: Path) -> None:
    """plot_mean_pre_dist writes one panel per layer to a non-empty file (spec §5)."""
    out = tmp_path / "mean_pre_dist.png"
    result = plots.plot_mean_pre_dist(_toy_stats(), out)

    assert result == out
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_entropy_map_writes_nonempty_file(tmp_path: Path) -> None:
    """plot_entropy_map writes a non-empty figure of sorted H(q) per layer (spec §5)."""
    out = tmp_path / "entropy_map.png"
    result = plots.plot_entropy_map(_toy_stats(), out)

    assert result == out
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_per_layer_entropy_writes_nonempty_file(tmp_path: Path) -> None:
    """plot_per_layer_entropy writes a non-empty bar chart, one value per layer (§5)."""
    out = tmp_path / "per_layer_entropy.png"
    result = plots.plot_per_layer_entropy(_toy_stats(), out)

    assert result == out
    assert out.exists()
    assert out.stat().st_size > 0
