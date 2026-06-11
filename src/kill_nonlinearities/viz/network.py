"""Phase-3 figures: visualize the de-nonlinearized network.

Three views of where the nonlinearity lives after regularization + surgery:

- ``plot_mode_composition`` — per-site stacked composition (exact-dead /
  near-dead / switching / near-on / exact-on) from hard-q statistics.
- ``plot_spatial_q_map`` — per-channel ``side x side`` heatmaps of a conv
  site's per-position hard q (the spatial structure of sign-consistency).
- ``plot_input_filters`` — rows of a weight matrix rendered in input space
  (the surviving nonlinear units' filters, or a fully-folded affine map's
  per-class templates).

Headless rendering rule as in ``viz.plots``: Agg before pyplot, no ``show``,
``savefig`` then ``close``; every function takes a full file path and returns it.
"""

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # must precede the pyplot import below

import matplotlib.pyplot as plt
from torch import Tensor

from kill_nonlinearities.analysis.statistics import NeuronStats

__all__ = [
    "plot_input_filters",
    "plot_mode_composition",
    "plot_spatial_q_map",
]

_CATEGORIES = (
    ("exact dead", "black"),
    ("near dead", "tab:blue"),
    ("switching", "tab:orange"),
    ("near on", "tab:green"),
    ("exact on", "gold"),
)


def _categorize(stats: Sequence[NeuronStats], low_entropy: float) -> dict[str, int]:
    counts = dict.fromkeys((name for name, _ in _CATEGORIES), 0)
    for s in stats:
        if s.q == 0.0:
            key = "exact dead"
        elif s.q == 1.0:
            key = "exact on"
        elif s.entropy > low_entropy:
            key = "switching"
        elif s.q < 0.5:
            key = "near dead"
        else:
            key = "near on"
        counts[key] += 1
    return counts


def plot_mode_composition(
    stats: Sequence[NeuronStats],
    low_entropy: float,
    path: Path,
    title: str = "Per-site nonlinearity composition",
) -> Path:
    """Stacked horizontal bars: each site's dead/on/switching fractions."""
    sites: dict[str, list[NeuronStats]] = {}
    for s in stats:
        sites.setdefault(s.site, []).append(s)

    fig, ax = plt.subplots(figsize=(8, 1.2 + 0.6 * len(sites)))
    names = list(sites)
    lefts = [0.0] * len(names)
    for category, color in _CATEGORIES:
        fractions = [
            _categorize(sites[name], low_entropy)[category] / len(sites[name])
            for name in names
        ]
        ax.barh(names, fractions, left=lefts, color=color, label=category)
        lefts = [left + f for left, f in zip(lefts, fractions, strict=True)]
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("fraction of units")
    ax.invert_yaxis()  # first site on top
    ax.set_title(f"{title} (H ≤ {low_entropy:g} nats)")
    ax.legend(fontsize=8, ncols=len(_CATEGORIES), loc="upper center")
    fig.tight_layout()
    try:
        fig.savefig(path, dpi=110, bbox_inches="tight")
    finally:
        plt.close(fig)
    return path


def plot_spatial_q_map(
    q: Tensor,
    side: int,
    path: Path,
    max_channels: int = 16,
    title: str = "Per-position hard q",
) -> Path:
    """Heatmap grid of a conv site's per-position q, one panel per channel.

    ``q`` is the site's per-position hard fraction-positive, either flat
    ``[C*side*side]`` in the emission convention ``i = (c*side + h)*side + w``
    or already shaped ``[C, side, side]``. The first ``max_channels`` channels
    are rendered with a shared 0..1 color scale.
    """
    grid = q.detach().reshape(-1, side, side)
    channels = min(int(grid.shape[0]), max_channels)
    cols = min(channels, 8)
    rows = (channels + cols - 1) // cols
    fig, axes = plt.subplots(
        rows, cols, figsize=(1.6 * cols, 1.8 * rows), squeeze=False
    )
    image = None
    for idx in range(rows * cols):
        ax = axes[idx // cols][idx % cols]
        ax.set_xticks(())
        ax.set_yticks(())
        if idx >= channels:
            ax.axis("off")
            continue
        image = ax.imshow(grid[idx].cpu(), vmin=0.0, vmax=1.0, cmap="viridis")
        ax.set_title(f"ch {idx}", fontsize=8)
    fig.suptitle(f"{title} (0 = always-negative, 1 = always-positive)")
    if image is not None:
        fig.colorbar(image, ax=axes, shrink=0.8, label="hard q")
    try:
        fig.savefig(path, dpi=110, bbox_inches="tight")
    finally:
        plt.close(fig)
    return path


def plot_input_filters(
    weight: Tensor,
    image_shape: tuple[int, ...],
    path: Path,
    max_filters: int = 32,
    title: str = "Input-space filters",
) -> Path:
    """Render rows of ``weight`` as input-space images.

    Grayscale ``(H, W)`` shapes use a diverging map with symmetric limits per
    filter (weights are signed); ``(3, H, W)`` shapes are min-max normalized
    per filter and shown as RGB.
    """
    rows_total = min(int(weight.shape[0]), max_filters)
    if rows_total == 0:
        raise ValueError("weight has no rows to render")
    cols = min(rows_total, 8)
    rows = (rows_total + cols - 1) // cols
    fig, axes = plt.subplots(
        rows, cols, figsize=(1.6 * cols, 1.8 * rows), squeeze=False
    )
    flat = weight.detach().cpu()
    for idx in range(rows * cols):
        ax = axes[idx // cols][idx % cols]
        ax.set_xticks(())
        ax.set_yticks(())
        if idx >= rows_total:
            ax.axis("off")
            continue
        image = flat[idx].reshape(image_shape)
        if len(image_shape) == 3 and image_shape[0] == 3:
            lo, hi = float(image.min()), float(image.max())
            scale = hi - lo if hi > lo else 1.0
            ax.imshow(((image - lo) / scale).permute(1, 2, 0))
        else:
            bound = float(image.abs().max()) or 1.0
            ax.imshow(image, cmap="RdBu_r", vmin=-bound, vmax=bound)
        ax.set_title(str(idx), fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    try:
        fig.savefig(path, dpi=110, bbox_inches="tight")
    finally:
        plt.close(fig)
    return path
