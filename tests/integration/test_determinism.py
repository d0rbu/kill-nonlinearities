"""Determinism: identical configs in one process give bit-identical results (§8)."""

from pathlib import Path

import torch

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


def _config(tmp_path: Path, sub: str) -> ExperimentConfig:
    """Identical config except for the output sub-directory (so artifacts don't clash)."""
    return ExperimentConfig(
        name=sub,
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
        checkpoint=CheckpointConfig(every_epochs=1, dir=str(tmp_path / sub)),
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


def test_two_identical_runs_are_bit_identical(tmp_path: Path) -> None:
    """Two same-config runs in one process give torch.equal history losses and q_i."""
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    try:
        result_a = run_experiment(_config(tmp_path, "run_a"), logger=InMemoryLogger())
        result_b = run_experiment(_config(tmp_path, "run_b"), logger=InMemoryLogger())
    finally:
        torch.use_deterministic_algorithms(False)

    total_a = torch.tensor([m.total_loss for m in result_a.train_result.history])
    total_b = torch.tensor([m.total_loss for m in result_b.train_result.history])
    assert torch.equal(total_a, total_b)

    task_a = torch.tensor([m.task_loss for m in result_a.train_result.history])
    task_b = torch.tensor([m.task_loss for m in result_b.train_result.history])
    assert torch.equal(task_a, task_b)

    reg_a = torch.tensor([m.reg_loss for m in result_a.train_result.history])
    reg_b = torch.tensor([m.reg_loss for m in result_b.train_result.history])
    assert torch.equal(reg_a, reg_b)

    q_a = torch.tensor([s.q for s in result_a.neuron_stats])
    q_b = torch.tensor([s.q for s in result_b.neuron_stats])
    assert torch.equal(q_a, q_b)
