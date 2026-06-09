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
from torch import Tensor

from kill_nonlinearities.analysis.statistics import NeuronStats
from kill_nonlinearities.surgery.apply import KPoint
from kill_nonlinearities.training.trainer import StepMetrics

__all__ = [
    "plot_acc_vs_k",
    "plot_entropy_map",
    "plot_loss_curves",
    "plot_mean_pre_dist",
    "plot_per_layer_entropy",
    "plot_soft_vs_hard",
]


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
    try:
        fig.savefig(path)
    finally:
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
    try:
        fig.savefig(path)
    finally:
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
    try:
        fig.savefig(path)
    finally:
        plt.close(fig)
    return path


def plot_per_layer_entropy(stats: Sequence[NeuronStats], path: Path) -> Path:
    """Bar chart of mean sign-entropy per layer (one value per site, spec §5)."""
    sites = _sites_in_order(stats)
    means = [
        sum(s.entropy for s in stats if s.site == site)
        / sum(1 for s in stats if s.site == site)
        for site in sites
    ]
    fig, ax = plt.subplots()
    ax.bar(sites, means)
    ax.set_xlabel("layer")
    ax.set_ylabel("mean H(q) [nats]")
    ax.set_title("Per-layer mean sign-entropy")
    try:
        fig.savefig(path)
    finally:
        plt.close(fig)
    return path


def _build_acc_vs_k_axes(
    k_points: Sequence[KPoint],
    random_k_points: Sequence[KPoint],
    total: int,
    lossless_prefix: int,
) -> tuple[plt.Figure, plt.Axes]:
    """Build (without saving) the acc-vs-k figure; shared by plot + tests (spec §7).

    ``total`` is guaranteed >= 1 by the caller (``make_k_grid`` always includes
    ``total``), so the ``k/total`` secondary-axis transform never divides by zero.
    """
    ks = [p.k for p in k_points]
    fig, ax = plt.subplots()
    ax.plot(ks, [p.val_acc for p in k_points], marker="o", label="val (entropy order)")
    ax.plot(
        ks, [p.test_acc for p in k_points], marker="s", label="test (entropy order)"
    )
    ax.plot(
        [p.k for p in random_k_points],
        [p.val_acc for p in random_k_points],
        marker="x",
        linestyle="--",
        label="val (random baseline)",
    )
    ax.axvline(lossless_prefix, color="grey", linestyle=":")
    ax.annotate(
        f"lossless prefix = {lossless_prefix}",
        xy=(lossless_prefix, ax.get_ylim()[0]),
        xytext=(4, 4),
        textcoords="offset points",
    )
    ax.set_xlabel("k (neurons converted)")
    ax.set_ylabel("accuracy")
    ax.set_title("Accuracy vs surgery k")
    ax.legend()

    secax = ax.secondary_xaxis(
        "top",
        functions=(lambda k: k / total, lambda frac: frac * total),
    )
    secax.set_xlabel("k / total")
    return fig, ax


def plot_acc_vs_k(
    k_points: Sequence[KPoint],
    random_k_points: Sequence[KPoint],
    total: int,
    lossless_prefix: int,
    path: Path,
) -> Path:
    """Accuracy vs k: entropy-order val/test + random baseline overlay (spec §5).

    A secondary top x-axis shows the linearized fraction ``k/total``; a vertical
    marker at the integer ``lossless_prefix`` (count of selection-set neurons with
    q exactly in {0, 1}) is annotated on the val curve only ([R17]). Takes a full
    file ``path`` and returns it after saving.
    """
    fig, _ = _build_acc_vs_k_axes(k_points, random_k_points, total, lossless_prefix)
    try:
        fig.savefig(path)
    finally:
        plt.close(fig)
    return path


def plot_soft_vs_hard(soft_p: Tensor, hard_q: Tensor, path: Path) -> Path:
    """Scatter of soft p_i (final tau) vs hard q_i with the y=x line (spec §5, [R3]).

    Validates that tau-annealing delivered hard sign consistency. Takes a full
    file ``path`` and returns it after saving.
    """
    soft = soft_p.detach().cpu().tolist()
    hard = hard_q.detach().cpu().tolist()
    fig, ax = plt.subplots()
    ax.scatter(hard, soft, alpha=0.6)
    ax.plot([0.0, 1.0], [0.0, 1.0], color="grey", linestyle="--", label="y = x")
    ax.set_xlabel("hard q_i  (z > 0 fraction)")
    ax.set_ylabel("soft p_i  (sigmoid(z / tau) mean)")
    ax.set_title("Soft vs hard fraction-positive")
    ax.legend()
    try:
        fig.savefig(path)
    finally:
        plt.close(fig)
    return path
