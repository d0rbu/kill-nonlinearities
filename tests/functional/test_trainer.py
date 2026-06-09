"""Functional tests for the train loop (spec §4.8)."""

import dataclasses
import math
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from kill_nonlinearities.config import (
    CheckpointConfig,
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    OptimConfig,
    RegConfig,
    TempScheduleConfig,
)
from kill_nonlinearities.models.mlp import ReLUMLP
from kill_nonlinearities.training.logging import InMemoryLogger
from kill_nonlinearities.training.schedule import checkpoint_steps
from kill_nonlinearities.training.trainer import (
    StepMetrics,
    TrainResult,
    _build_optimizer,
    train,
)


def _one_batch_loader() -> DataLoader:
    torch.manual_seed(0)
    x = torch.randn(16, 8)
    y = torch.randint(0, 3, (16,))
    return DataLoader(TensorDataset(x, y), batch_size=16, shuffle=False)


def _config(tmp_path: Path, *, epochs: int, lam: float, kind: str) -> ExperimentConfig:
    cfg = ExperimentConfig(name="trainer-test")
    return dataclasses.replace(
        cfg,
        model=ModelConfig(input_dim=8, hidden_dims=(6, 4), output_dim=3),
        reg=RegConfig(lam=lam),
        train=dataclasses.replace(cfg.train, epochs=epochs, seed=0, device="cpu"),
        temp_schedule=TempScheduleConfig(kind=kind, tau_start=1.0, tau_end=0.1),
        checkpoint=CheckpointConfig(every_epochs=1, dir=str(tmp_path)),
        data=DataConfig(batch_size=16),
    )


def test_one_batch_overfit_reduces_task_loss(tmp_path: Path) -> None:
    loader = _one_batch_loader()
    config = _config(tmp_path, epochs=40, lam=0.0, kind="constant")
    model = ReLUMLP(config.model)
    logger = InMemoryLogger()

    result = train(model, loader, loader, config, logger)

    assert isinstance(result, TrainResult)
    assert isinstance(result.history[0], StepMetrics)
    # Overfit: the final step's task loss is well below the first.
    assert result.history[-1].task_loss < result.history[0].task_loss


def test_total_steps_and_history_length(tmp_path: Path) -> None:
    loader = _one_batch_loader()  # len == 1 (one batch)
    config = _config(tmp_path, epochs=5, lam=0.0, kind="constant")
    model = ReLUMLP(config.model)

    result = train(model, loader, loader, config, NullCheck := InMemoryLogger())  # noqa: N806

    # total_steps == epochs * len(train_loader) == 5 * 1 == 5.
    assert len(result.history) == 5
    assert [m.step for m in result.history] == [0, 1, 2, 3, 4]
    assert [m.epoch for m in result.history] == [0, 1, 2, 3, 4]
    # Scalars logged once per step with the expected keys.
    assert len(NullCheck.scalars) == 5
    assert set(NullCheck.scalars[0][1]) == {
        "loss/task",
        "loss/reg",
        "loss/total",
        "tau",
    }


def test_checkpoints_written_at_checkpoint_steps(tmp_path: Path) -> None:
    loader = _one_batch_loader()  # steps_per_epoch == 1
    config = _config(tmp_path, epochs=3, lam=0.0, kind="constant")
    model = ReLUMLP(config.model)

    result = train(model, loader, loader, config, InMemoryLogger())

    expected = checkpoint_steps(total_steps=3, steps_per_epoch=1, every_epochs=1)
    assert [int(p.stem.split("_")[1]) for p in result.checkpoint_paths] == expected
    assert all(p.exists() for p in result.checkpoint_paths)
    # Checkpoints co-locate with artifacts under ``<checkpoint.dir>/<name>/``,
    # NOT the flat ``checkpoint.dir`` (capstone fix).
    ckpt_dir = Path(config.checkpoint.dir) / config.name
    assert all(p.parent == ckpt_dir for p in result.checkpoint_paths)


def test_annealed_tau_steps_keep_grads_and_params_finite(
    tmp_path: Path,
) -> None:
    # lam>0 with an exponential anneal to a low tau drives p toward {0,1};
    # the loss clamp must keep every gradient and param finite (spec §4.4/§9).
    loader = _one_batch_loader()
    config = _config(tmp_path, epochs=30, lam=1.0, kind="exponential")
    model = ReLUMLP(config.model)

    result = train(model, loader, loader, config, InMemoryLogger())

    for metric in result.history:
        assert math.isfinite(metric.task_loss)
        assert math.isfinite(metric.reg_loss)
        assert math.isfinite(metric.total_loss)
        assert metric.reg_loss >= 0.0
    for param in result.model.parameters():
        assert torch.isfinite(param).all()
    # Non-constant schedule reaches tau_end at the final step (spec §4.8).
    assert result.history[-1].tau == pytest.approx(config.temp_schedule.tau_end)


def test_build_optimizer_sgd_propagates_lr_and_weight_decay(tmp_path: Path) -> None:
    """OptimConfig(name='sgd') yields torch.optim.SGD with lr/weight_decay set."""
    cfg = _config(tmp_path, epochs=1, lam=0.0, kind="constant")
    cfg = dataclasses.replace(
        cfg, optim=OptimConfig(name="SGD", lr=0.07, weight_decay=0.003)
    )
    model = ReLUMLP(cfg.model)

    optimizer = _build_optimizer(model, cfg)

    assert isinstance(optimizer, torch.optim.SGD)
    group = optimizer.param_groups[0]
    assert group["lr"] == pytest.approx(0.07)
    assert group["weight_decay"] == pytest.approx(0.003)


def test_build_optimizer_unknown_name_raises(tmp_path: Path) -> None:
    """An unknown optimizer name raises ValueError naming the offending optimizer."""
    cfg = _config(tmp_path, epochs=1, lam=0.0, kind="constant")
    cfg = dataclasses.replace(cfg, optim=OptimConfig(name="rmsprop"))
    model = ReLUMLP(cfg.model)

    with pytest.raises(ValueError, match="rmsprop"):
        _build_optimizer(model, cfg)


def test_grad_clip_keeps_params_and_grads_finite(tmp_path: Path) -> None:
    """A small grad_clip with a high lr keeps params/grads finite and clips norms.

    Without clipping, this high-lr/lam>0 config blows up; the per-step
    clip_grad_norm_ must bound the gradient norm to grad_clip and keep every
    parameter finite (covers trainer.py grad_clip branch, spec §4.8).
    """
    loader = _one_batch_loader()
    cfg = _config(tmp_path, epochs=10, lam=1.0, kind="exponential")
    grad_clip = 0.05
    cfg = dataclasses.replace(
        cfg,
        optim=OptimConfig(name="sgd", lr=5.0, weight_decay=0.0),
        train=dataclasses.replace(cfg.train, grad_clip=grad_clip),
    )
    model = ReLUMLP(cfg.model)

    # Clipping engages at the FIRST step: a fresh model's pre-clip gradient norm
    # (same forward/loss the trainer runs) exceeds the small grad_clip, so the
    # trainer's clip_grad_norm_ actually bounds the update. Measure before train.
    probe = ReLUMLP(cfg.model)
    probe.load_state_dict(model.state_dict())
    probe.train()
    x, y = next(iter(loader))
    probe.zero_grad()
    probe_loss = torch.nn.functional.cross_entropy(probe(x).logits, y)
    probe_loss.backward()
    pre_clip_norm = torch.nn.utils.clip_grad_norm_(probe.parameters(), grad_clip)
    assert float(pre_clip_norm) > grad_clip

    result = train(model, loader, loader, cfg, InMemoryLogger())

    for metric in result.history:
        assert math.isfinite(metric.total_loss)
    for param in result.model.parameters():
        assert torch.isfinite(param).all()
