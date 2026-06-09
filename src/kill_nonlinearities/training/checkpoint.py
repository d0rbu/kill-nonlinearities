"""Checkpoint save/load (spec §4.7).

A checkpoint stores the model ``state_dict`` (including ``SelectiveReLU.mode``
buffers), the training ``step``, and the serialized ``ExperimentConfig``.
``load_checkpoint`` does a **strict** load into an architecturally-identical
model; callers rebuild a fresh ``ReLUMLP`` from the checkpoint's serialized
``ModelConfig`` before loading.
"""

from pathlib import Path

import torch
from torch import nn

from kill_nonlinearities.config import ExperimentConfig


def save_checkpoint(
    model: nn.Module, step: int, config: ExperimentConfig, dir: Path
) -> Path:
    """Write ``state_dict`` + ``step`` + ``config`` to ``dir/step_{step}.pt``."""
    directory = Path(dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"step_{step}.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "step": step,
            "config": config,
        },
        path,
    )
    return path


def load_checkpoint(path: Path, model: nn.Module) -> int:
    """Strictly load ``path`` into ``model`` and return the saved ``step``."""
    blob = torch.load(Path(path), weights_only=False)
    model.load_state_dict(blob["state_dict"], strict=True)
    return int(blob["step"])
