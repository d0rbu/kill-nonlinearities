"""Integration: run_experiment end-to-end on a tiny synthetic ReLUCNN (conv spec 2026-06-10).

Mirrors the synthetic ReLUMLP integration config; the synthetic provider emits
[in_channels, image_size, image_size] inputs for kind="cnn" and the probe batch
keeps its native shape, so the full train -> analyze -> surgery -> viz pipeline
runs offline with conv sites.
"""

import dataclasses
import math
from pathlib import Path

from kill_nonlinearities.analysis.selection import make_k_grid
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
from kill_nonlinearities.experiments.run import run_experiment
from kill_nonlinearities.training.logging import InMemoryLogger
from kill_nonlinearities.viz.animation import gif_frame_count


def make_cnn_config(tmp_path: Path) -> ExperimentConfig:
    """Tiny, seeded, lambda>0 synthetic CNN config writing artifacts under tmp_path."""
    return ExperimentConfig(
        name="it-cnn",
        model=ModelConfig(
            input_dim=1,  # ignored for kind="cnn"
            hidden_dims=(6,),
            output_dim=3,
            kind="cnn",
            in_channels=2,
            image_size=8,
            conv_channels=(3, 4),
        ),
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


def test_run_experiment_cnn_full_pipeline(tmp_path: Path) -> None:
    """The full pipeline runs on conv sites: stats, k-sweep, and artifacts all land."""
    config = make_cnn_config(tmp_path)
    result = run_experiment(config, logger=InMemoryLogger())

    # Conv sites then FC sites, one neuron per (c, h, w) position: 3*8*8, 4*4*4, 6.
    expected_sites = ("conv0", "conv1", "fc0")
    assert result.model.site_names == expected_sites
    site_counts = dict.fromkeys(expected_sites, 0)
    for s in result.neuron_stats:
        site_counts[s.site] += 1
    assert site_counts == {"conv0": 192, "conv1": 64, "fc0": 6}
    assert tuple(result.frames[0].q_by_site) == expected_sites

    # Training ran and is finite.
    assert result.train_result.history
    for m in result.train_result.history:
        assert math.isfinite(m.task_loss)
        assert math.isfinite(m.reg_loss)

    # k-sweep covers the grid over all 262 neurons, val+test in range.
    total = len(result.neuron_stats)
    assert total == 262
    expected_grid = make_k_grid(total, config.surgery.num_k)
    assert [kp.k for kp in result.k_points] == expected_grid
    assert [kp.k for kp in result.random_k_points] == expected_grid
    for kp in (*result.k_points, *result.random_k_points):
        assert 0.0 <= kp.val_acc <= 1.0
        assert 0.0 <= kp.test_acc <= 1.0

    # All artifacts render, and the GIFs have one frame per checkpoint.
    assert len(result.artifact_paths) == 8
    for key, path in result.artifact_paths.items():
        assert path.is_file(), key
        assert path.stat().st_size > 0, key
    n_checkpoints = len(result.train_result.checkpoint_paths)
    assert len(result.frames) == n_checkpoints
    assert gif_frame_count(result.artifact_paths["activation_gif"]) == n_checkpoints
    assert gif_frame_count(result.artifact_paths["qi_bimodality_gif"]) == n_checkpoints


def test_run_experiment_cnn_channel_granularity(tmp_path: Path) -> None:
    """The channel-pooled regularizer variant trains end-to-end with finite losses.

    Analysis/surgery stay per-position (same neuron count and k-grid as the
    per-neuron run); only the training incentive changes (conv spec addendum).
    """
    base = make_cnn_config(tmp_path)
    config = dataclasses.replace(
        base,
        name="it-cnn-chanreg",
        reg=dataclasses.replace(base.reg, granularity="channel"),
    )

    result = run_experiment(config, logger=InMemoryLogger())

    assert result.train_result.history
    for m in result.train_result.history:
        assert math.isfinite(m.task_loss)
        assert math.isfinite(m.reg_loss)
    # Same per-position measurement axis as the per-neuron run.
    assert len(result.neuron_stats) == 262
    assert [kp.k for kp in result.k_points] == make_k_grid(262, config.surgery.num_k)


def test_cnn_pre_activation_columns_are_per_position_neurons(
    tmp_path: Path,
) -> None:
    """Conv sites emit [B, C*H*W]: rows are batch samples, one column per position."""
    import torch

    config = make_cnn_config(tmp_path)
    result = run_experiment(config, logger=InMemoryLogger())

    model = result.model
    model.eval()
    x = torch.randn(2, 2, 8, 8)
    with torch.no_grad():
        out = model(x)
    assert out.pre_activations[0].shape == (2, 3 * 8 * 8)
    assert out.pre_activations[1].shape == (2, 4 * 4 * 4)
    assert out.pre_activations[2].shape == (2, 6)
