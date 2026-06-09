"""Unit tests for kill_nonlinearities.viz.plots (Agg headless figures)."""

from pathlib import Path

import matplotlib

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
