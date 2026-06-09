"""Integration tests for run_experiment (spec §4.13, §5, §7)."""

import importlib


def test_experiments_package_imports() -> None:
    """The experiments package is importable as a namespace marker."""
    module = importlib.import_module("kill_nonlinearities.experiments")
    assert module is not None
