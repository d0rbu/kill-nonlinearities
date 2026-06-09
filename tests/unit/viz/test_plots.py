"""Unit tests for kill_nonlinearities.viz.plots (Agg headless figures)."""

import matplotlib


def test_importing_plots_selects_agg_backend() -> None:
    """Importing viz.plots forces the non-interactive Agg backend (spec §4.12)."""
    import kill_nonlinearities.viz.plots  # noqa: F401

    assert matplotlib.get_backend().lower() == "agg"
