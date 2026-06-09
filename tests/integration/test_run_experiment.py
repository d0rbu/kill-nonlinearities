"""Integration tests for run_experiment (spec §4.13, §5, §7)."""

import dataclasses
import importlib
import math
from pathlib import Path

import pytest

from kill_nonlinearities.config import (
    CheckpointConfig,
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    OptimConfig,
    ProbeConfig,
    RegConfig,
    SurgeryConfig,
    TempScheduleConfig,
    TrainConfig,
    WandbConfig,
)
from kill_nonlinearities.experiments.run import ExperimentResult, run_experiment
from kill_nonlinearities.training.logging import InMemoryLogger


def make_synthetic_config(tmp_path: Path) -> ExperimentConfig:
    """Tiny, seeded, lambda>0 synthetic config writing all artifacts under tmp_path."""
    return ExperimentConfig(
        name="it-synthetic",
        model=ModelConfig(input_dim=12, hidden_dims=(8, 8), output_dim=3),
        optim=OptimConfig(lr=1e-2, weight_decay=0.0, name="adam"),
        temp_schedule=TempScheduleConfig(
            kind="exponential", tau_start=1.0, tau_end=0.1
        ),
        reg=RegConfig(lam=0.05, entropy_eps=1e-6),
        data=DataConfig(
            dataset="synthetic",
            batch_size=8,
            eval_batch_size=16,
            val_fraction=0.25,
            split_seed=0,
            drop_last=False,
            data_dir=str(tmp_path / "data"),
            num_workers=0,
        ),
        probe=ProbeConfig(num_neurons=4, seed=0, batch_size=16),
        checkpoint=CheckpointConfig(every_epochs=1, dir=str(tmp_path / "runs")),
        surgery=SurgeryConfig(num_k=5, random_baseline=True, tie_break="identity"),
        wandb=WandbConfig(
            project="kill-nonlinearities",
            entity=None,
            mode="disabled",
            group=None,
            tags=(),
        ),
        train=TrainConfig(epochs=2, seed=0, device="cpu", grad_clip=None),
    )


@pytest.fixture
def synthetic_config(tmp_path: Path) -> ExperimentConfig:
    return make_synthetic_config(tmp_path)


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


def test_run_experiment_returns_populated_result(
    synthetic_config: ExperimentConfig,
) -> None:
    """run_experiment wires the full pipeline and returns a populated result (§5)."""
    result = run_experiment(synthetic_config, logger=InMemoryLogger())

    assert isinstance(result, ExperimentResult)
    assert result.train_result.history
    for m in result.train_result.history:
        assert math.isfinite(m.task_loss)
        assert math.isfinite(m.reg_loss)
        assert math.isfinite(m.total_loss)
    assert result.neuron_stats
    assert result.k_points
    assert result.random_k_points
    assert result.frames
    assert set(result.artifact_paths) == {
        "loss_curves",
        "mean_pre_dist",
        "entropy_map",
        "per_layer_entropy",
        "acc_vs_k",
        "soft_vs_hard",
        "activation_gif",
        "qi_bimodality_gif",
    }
