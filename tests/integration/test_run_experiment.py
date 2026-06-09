"""Integration tests for run_experiment (spec §4.13, §5, §7)."""

import copy
import dataclasses
import importlib
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from kill_nonlinearities.analysis.selection import (
    assign_modes,
    make_k_grid,
    rank_by_entropy,
    select_topk,
)
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
from kill_nonlinearities.data.datasets import make_dataloaders
from kill_nonlinearities.experiments.run import (
    ExperimentResult,
    config_from_json,
    run_experiment,
)
from kill_nonlinearities.models.activations import ActivationMode
from kill_nonlinearities.surgery.apply import apply_modes
from kill_nonlinearities.training.logging import InMemoryLogger
from kill_nonlinearities.training.schedule import checkpoint_steps
from kill_nonlinearities.viz import plots as viz_plots
from kill_nonlinearities.viz.animation import gif_frame_count


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


def test_checkpoints_land_at_expected_steps(
    synthetic_config: ExperimentConfig,
) -> None:
    """Checkpoint count/positions match checkpoint_steps for the realized total_steps."""
    result = run_experiment(synthetic_config, logger=InMemoryLogger())

    train_loader, _, _ = make_dataloaders(synthetic_config)
    steps_per_epoch = len(train_loader)
    total_steps = synthetic_config.train.epochs * steps_per_epoch
    expected = checkpoint_steps(
        total_steps, steps_per_epoch, synthetic_config.checkpoint.every_epochs
    )

    assert len(result.train_result.checkpoint_paths) == len(expected)
    assert all(p.exists() for p in result.train_result.checkpoint_paths)


def test_k_sweep_covers_every_grid_point_with_val_and_test(
    synthetic_config: ExperimentConfig,
) -> None:
    """Each KPoint carries a finite val_acc and test_acc, one per unique k-grid point."""
    result = run_experiment(synthetic_config, logger=InMemoryLogger())

    total = len(result.neuron_stats)
    expected_grid = make_k_grid(total, synthetic_config.surgery.num_k)

    assert [kp.k for kp in result.k_points] == expected_grid
    assert [kp.k for kp in result.random_k_points] == expected_grid
    for kp in (*result.k_points, *result.random_k_points):
        assert 0.0 <= kp.val_acc <= 1.0
        assert 0.0 <= kp.test_acc <= 1.0
        assert math.isfinite(kp.val_acc)
        assert math.isfinite(kp.test_acc)


def test_run_logs_val_acc_sweep_metric(
    synthetic_config: ExperimentConfig,
) -> None:
    """run_experiment logs the 'val/acc' metric the sweep config maximizes (§4.13)."""
    logger = InMemoryLogger()
    run_experiment(synthetic_config, logger=logger)

    logged_keys = {k for _, values in logger.scalars for k in values}
    assert "val/acc" in logged_keys
    val_acc_values = [
        values["val/acc"] for _, values in logger.scalars if "val/acc" in values
    ]
    assert len(val_acc_values) == 1
    assert 0.0 <= val_acc_values[0] <= 1.0


def test_all_artifact_files_exist_and_are_non_empty(
    synthetic_config: ExperimentConfig,
) -> None:
    """Every rendered artifact is a real, non-empty file on disk."""
    result = run_experiment(synthetic_config, logger=InMemoryLogger())

    assert len(result.artifact_paths) == 8
    for key, path in result.artifact_paths.items():
        assert isinstance(path, Path), key
        assert path.is_file(), key
        assert path.stat().st_size > 0, key


def test_both_gifs_have_one_frame_per_checkpoint(
    synthetic_config: ExperimentConfig,
) -> None:
    """activation_gif and qi_bimodality_gif each have len(checkpoint_steps) frames."""
    result = run_experiment(synthetic_config, logger=InMemoryLogger())

    train_loader, _, _ = make_dataloaders(synthetic_config)
    steps_per_epoch = len(train_loader)
    total_steps = synthetic_config.train.epochs * steps_per_epoch
    n_checkpoints = len(
        checkpoint_steps(
            total_steps, steps_per_epoch, synthetic_config.checkpoint.every_epochs
        )
    )

    assert len(result.frames) == n_checkpoints
    assert gif_frame_count(result.artifact_paths["activation_gif"]) == n_checkpoints
    assert gif_frame_count(result.artifact_paths["qi_bimodality_gif"]) == n_checkpoints


def test_acc_vs_k_axes_data_equals_k_points(
    synthetic_config: ExperimentConfig,
) -> None:
    """The acc-vs-k figure's val/test line ydata equals the k_sweep accuracies ([R23])."""
    result = run_experiment(synthetic_config, logger=InMemoryLogger())

    total = len(result.neuron_stats)
    lossless_prefix = sum(1 for s in result.neuron_stats if s.q in (0.0, 1.0))
    fig, ax = viz_plots._build_acc_vs_k_axes(
        result.k_points, result.random_k_points, total, lossless_prefix
    )
    try:
        lines = ax.get_lines()
        val_line = next(
            line for line in lines if line.get_label() == "val (entropy order)"
        )
        test_line = next(
            line for line in lines if line.get_label() == "test (entropy order)"
        )
        assert list(val_line.get_xdata()) == [  # ty: ignore[invalid-argument-type]
            kp.k for kp in result.k_points
        ]
        assert list(val_line.get_ydata()) == [  # ty: ignore[invalid-argument-type]
            kp.val_acc for kp in result.k_points
        ]
        assert list(test_line.get_ydata()) == [  # ty: ignore[invalid-argument-type]
            kp.test_acc for kp in result.k_points
        ]
    finally:
        viz_plots.plt.close(fig)


def test_k_equals_total_converts_all_neurons_and_logits_finite(
    synthetic_config: ExperimentConfig,
) -> None:
    """At k == total every neuron is converted (no RELU left) and logits are finite."""
    result = run_experiment(synthetic_config, logger=InMemoryLogger())

    model = result.model
    stats = result.neuron_stats
    total = len(stats)
    ranked = rank_by_entropy(stats)

    work = copy.deepcopy(model)
    widths = {
        site: int(act.mode.shape[0])
        for site, act in zip(model.site_names, model.activations, strict=True)
    }
    modes = assign_modes(
        select_topk(ranked, total), tie_break="identity", widths=widths
    )
    apply_modes(work, modes)

    # Every neuron has been converted away from RELU (no mode == RELU remains).
    relu = int(ActivationMode.RELU)
    for act in work.activations:
        assert not bool((act.mode == relu).any())

    # Logits over a probe input are finite.
    probe = torch.randn(5, synthetic_config.model.input_dim)
    work.eval()
    with torch.no_grad():
        logits = work(probe).logits
    assert torch.isfinite(logits).all()


def test_config_from_json_builds_nested_config(tmp_path: Path) -> None:
    """config_from_json maps a nested JSON object into an ExperimentConfig (§4.13)."""
    payload = {
        "name": "from-json",
        "model": {"input_dim": 12, "hidden_dims": [8, 8], "output_dim": 3},
        "optim": {"lr": 0.01, "weight_decay": 0.0, "name": "adam"},
        "temp_schedule": {"kind": "exponential", "tau_start": 1.0, "tau_end": 0.1},
        "reg": {"lam": 0.05, "entropy_eps": 1e-6},
        "data": {"dataset": "synthetic", "batch_size": 8},
        "surgery": {"num_k": 5, "random_baseline": True, "tie_break": "identity"},
        "train": {"epochs": 2, "seed": 0, "device": "cpu"},
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload))

    config = config_from_json(path)

    assert config.name == "from-json"
    assert config.model.input_dim == 12
    assert config.model.hidden_dims == (8, 8)  # JSON array coerced to tuple
    assert config.optim.lr == 0.01
    assert config.temp_schedule.kind == "exponential"
    assert config.reg.lam == 0.05
    assert config.data.dataset == "synthetic"
    assert config.data.batch_size == 8
    assert config.surgery.num_k == 5
    assert config.train.epochs == 2


def test_config_from_json_missing_section_uses_defaults(tmp_path: Path) -> None:
    """Sections absent from the JSON fall back to ExperimentConfig() defaults."""
    path = tmp_path / "minimal.json"
    path.write_text(json.dumps({"name": "minimal"}))

    config = config_from_json(path)
    default = ExperimentConfig()

    assert config.name == "minimal"
    assert config.model == default.model
    assert config.optim == default.optim
    assert config.train == default.train


def test_config_from_json_coerces_wandb_tags_and_default_hidden_dims(
    tmp_path: Path,
) -> None:
    """The wandb section's tags array becomes a tuple; model without hidden_dims works."""
    payload = {
        "model": {"input_dim": 12, "output_dim": 3},  # no hidden_dims -> default
        "wandb": {
            "project": "p",
            "entity": None,
            "mode": "disabled",
            "group": None,
            "tags": ["a", "b"],
        },
    }
    path = tmp_path / "wandb.json"
    path.write_text(json.dumps(payload))

    config = config_from_json(path)

    assert config.model.input_dim == 12
    assert config.model.hidden_dims == ExperimentConfig().model.hidden_dims
    assert config.wandb.tags == ("a", "b")  # JSON array coerced to tuple


def test_cli_module_help_runs() -> None:
    """`python -m kill_nonlinearities.experiments.run --help` exits 0 (§4.13 CLI)."""
    proc = subprocess.run(
        [sys.executable, "-m", "kill_nonlinearities.experiments.run", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "Run a Phase-1a experiment." in proc.stdout
