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

from kill_nonlinearities.analysis.statistics import NeuronStats
from kill_nonlinearities.training.trainer import StepMetrics

__all__ = ["plot_entropy_map", "plot_loss_curves", "plot_mean_pre_dist"]


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


def _sites_in_order(stats: Sequence[NeuronStats]) -> list[str]:
    """Distinct site ids in first-appearance (forward) order."""
    seen: dict[str, None] = {}
    for s in stats:
        seen.setdefault(s.site, None)
    return list(seen)


def plot_mean_pre_dist(stats: Sequence[NeuronStats], path: Path) -> Path:
    """Histogram of per-neuron mean pre-activation, one panel per layer (spec §5).

    Layers differ in scale, so each site gets its own subplot. Takes a full file
    ``path`` and returns it after saving.
    """
    sites = _sites_in_order(stats)
    fig, axes = plt.subplots(1, len(sites), squeeze=False)
    for ax, site in zip(axes[0], sites, strict=True):
        values = [s.mean_pre for s in stats if s.site == site]
        ax.hist(values, bins=min(len(values), 20))
        ax.set_title(site)
        ax.set_xlabel("mean pre-activation")
        ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_entropy_map(stats: Sequence[NeuronStats], path: Path) -> Path:
    """Per-layer sign-entropy sorted ascending, highlighting low/high H(q) (spec §5).

    One subplot per layer; bars sorted by entropy so the lowest-entropy (most
    eliminable) neurons sit on the left. Takes a full file ``path``; returns it.
    """
    sites = _sites_in_order(stats)
    fig, axes = plt.subplots(len(sites), 1, squeeze=False)
    for ax, site in zip(axes[:, 0], sites, strict=True):
        entropies = sorted(s.entropy for s in stats if s.site == site)
        ax.bar(range(len(entropies)), entropies)
        ax.set_title(f"{site}: sign-entropy (ascending)")
        ax.set_xlabel("neuron rank")
        ax.set_ylabel("H(q) [nats]")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path
