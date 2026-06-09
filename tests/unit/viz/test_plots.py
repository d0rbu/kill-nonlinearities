"""Unit tests for kill_nonlinearities.viz.plots (Agg headless figures)."""

from pathlib import Path

import matplotlib

from kill_nonlinearities.analysis.statistics import NeuronStats
from kill_nonlinearities.surgery.apply import KPoint
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


def _toy_kpoints() -> list[KPoint]:
    return [
        KPoint(k=0, val_acc=0.90, test_acc=0.88),
        KPoint(k=2, val_acc=0.85, test_acc=0.83),
        KPoint(k=4, val_acc=0.70, test_acc=0.68),
    ]


def _toy_random_kpoints() -> list[KPoint]:
    return [
        KPoint(k=0, val_acc=0.90, test_acc=0.88),
        KPoint(k=2, val_acc=0.60, test_acc=0.58),
        KPoint(k=4, val_acc=0.40, test_acc=0.38),
    ]


def test_plot_acc_vs_k_writes_nonempty_file(tmp_path: Path) -> None:
    """plot_acc_vs_k writes a non-empty figure with val/test + random overlay (§5)."""
    out = tmp_path / "acc_vs_k.png"
    result = plots.plot_acc_vs_k(
        _toy_kpoints(),
        _toy_random_kpoints(),
        total=4,
        lossless_prefix=2,
        path=out,
    )

    assert result == out
    assert out.exists()
    assert out.stat().st_size > 0
