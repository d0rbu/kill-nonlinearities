"""Integration tests for run_experiment (spec §4.13, §5, §7)."""

import dataclasses
import importlib

from kill_nonlinearities.experiments.run import ExperimentResult


def test_experiments_package_imports() -> None:
    """The experiments package is importable as a namespace marker."""
    module = importlib.import_module("kill_nonlinearities.experiments")
    assert module is not None


def test_experiment_result_has_expected_fields() -> None:
    """ExperimentResult exposes the spec §4.13 fields, including artifact_paths."""
    field_names = {f.name for f in dataclasses.fields(ExperimentResult)}
    assert field_names == {
        "model",
        "train_result",
        "neuron_stats",
        "k_points",
        "random_k_points",
        "frames",
        "artifact_paths",
    }
