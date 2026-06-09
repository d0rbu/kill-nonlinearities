"""Static figures for the Phase 1a pipeline (spec §4.12, §5).

Headless rendering rule (spec §4.12, [R10]): select the non-interactive Agg
backend *before* importing pyplot; never call ``plt.show``; always ``savefig``
then ``plt.close(fig)``. Every public function takes a full FILE path and returns
the saved figure ``Path``.
"""

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # must precede the pyplot import below (spec §4.12)

import matplotlib.pyplot as plt

from kill_nonlinearities.training.trainer import StepMetrics

__all__ = ["plot_loss_curves"]


def plot_loss_curves(history: Sequence[StepMetrics], path: Path) -> Path:
    """Plot task/reg/total loss vs step from training ``history`` (spec §5).

    Takes a full file ``path`` and returns it after saving.
    """
    steps = [m.step for m in history]
    fig, ax = plt.subplots()
    ax.plot(steps, [m.task_loss for m in history], label="task")
    ax.plot(steps, [m.reg_loss for m in history], label="reg")
    ax.plot(steps, [m.total_loss for m in history], label="total")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title("Training losses")
    ax.legend()
    fig.savefig(path)
    plt.close(fig)
    return path
