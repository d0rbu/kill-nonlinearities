"""The training loop (spec §4.8).

``total_steps = epochs * len(train_loader)`` is derived **after** the loaders
exist (it already reflects the val carve-out and ``drop_last``); the
``TemperatureSchedule`` is built here from that ``total_steps``. Each step runs
``task + lambda * reg``, appends a ``StepMetrics`` to ``history``, logs scalars,
and checkpoints at exactly ``checkpoint_steps(...)``.
"""

from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from kill_nonlinearities.config import ExperimentConfig
from kill_nonlinearities.models.base import PreActModel
from kill_nonlinearities.regularization.loss import (
    grouped_sign_consistency_loss,
    sign_consistency_loss,
)
from kill_nonlinearities.training.checkpoint import save_checkpoint
from kill_nonlinearities.training.logging import Logger, NullLogger
from kill_nonlinearities.training.schedule import (
    TemperatureSchedule,
    checkpoint_steps,
)


@dataclass(frozen=True)
class StepMetrics:
    step: int
    epoch: int
    task_loss: float
    reg_loss: float
    total_loss: float
    tau: float


@dataclass
class TrainResult:
    model: PreActModel
    history: list[StepMetrics] = field(default_factory=list)
    checkpoint_paths: list[Path] = field(default_factory=list)


def _build_optimizer(
    model: nn.Module, config: ExperimentConfig
) -> torch.optim.Optimizer:
    name = config.optim.name.lower()
    if name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=config.optim.lr,
            weight_decay=config.optim.weight_decay,
        )
    if name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=config.optim.lr,
            weight_decay=config.optim.weight_decay,
        )
    msg = f"unknown optimizer {config.optim.name!r}; expected 'adam' or 'sgd'"
    raise ValueError(msg)


def train(
    model: PreActModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: ExperimentConfig,
    logger: Logger | None = None,
) -> TrainResult:
    """Train ``model`` and return its history + written checkpoint paths."""
    logger = logger if logger is not None else NullLogger()
    model.to(config.train.device)

    steps_per_epoch = len(train_loader)
    total_steps = config.train.epochs * steps_per_epoch
    schedule = TemperatureSchedule(
        kind=config.temp_schedule.kind,
        tau_start=config.temp_schedule.tau_start,
        tau_end=config.temp_schedule.tau_end,
        total_steps=total_steps,
    )
    save_at = set(
        checkpoint_steps(
            total_steps=total_steps,
            steps_per_epoch=steps_per_epoch,
            every_epochs=config.checkpoint.every_epochs,
        )
    )

    optimizer = _build_optimizer(model, config)
    criterion = nn.CrossEntropyLoss()
    result = TrainResult(model=model)

    # Checkpoints co-locate with run artifacts under ``runs/<name>/`` so two runs
    # that share ``checkpoint.dir`` but differ in ``name`` never clobber each
    # other's ``step_N.pt`` (capstone fix).
    ckpt_dir = Path(config.checkpoint.dir) / config.name

    step = 0
    model.train()
    for epoch in range(config.train.epochs):
        for x, y in train_loader:
            x = x.to(config.train.device)
            y = y.to(config.train.device)
            tau = schedule(step)

            optimizer.zero_grad()
            out = model(x)
            task = criterion(out.logits, y)
            reg = (
                grouped_sign_consistency_loss(
                    out.pre_activations,
                    model.site_group_sizes,
                    tau,
                    config.reg.entropy_eps,
                )
                if config.reg.granularity == "channel"
                else sign_consistency_loss(
                    out.pre_activations, tau, config.reg.entropy_eps
                )
            )
            total = task + config.reg.lam * reg
            total.backward()
            if config.train.grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip)
            optimizer.step()

            metrics = StepMetrics(
                step=step,
                epoch=epoch,
                task_loss=float(task.detach()),
                reg_loss=float(reg.detach()),
                total_loss=float(total.detach()),
                tau=tau,
            )
            result.history.append(metrics)
            logger.log_scalars(
                {
                    "loss/task": metrics.task_loss,
                    "loss/reg": metrics.reg_loss,
                    "loss/total": metrics.total_loss,
                    "tau": metrics.tau,
                },
                step=step,
            )

            if step in save_at:
                path = save_checkpoint(
                    model,
                    step=step,
                    config=config,
                    dir=ckpt_dir,
                )
                result.checkpoint_paths.append(path)

            step += 1

    return result
