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

from collections.abc import Callable, Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # must precede the pyplot import below

import matplotlib.pyplot as plt
from torch import Tensor

from kill_nonlinearities.analysis.decompile import Branch, Leaf, Node, tree_stats
from kill_nonlinearities.analysis.statistics import NeuronStats

__all__ = [
    "plot_class_q_matrix",
    "plot_decision_tree",
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
    panel_labels: Sequence[str] | None = None,
) -> Path:
    """Heatmap grid of a conv site's per-position q, one panel per channel.

    ``q`` is the site's per-position hard fraction-positive, either flat
    ``[C*side*side]`` in the emission convention ``i = (c*side + h)*side + w``
    or already shaped ``[C, side, side]``. The first ``max_channels`` channels
    are rendered with a shared 0..1 color scale. ``panel_labels`` overrides the
    default ``ch {i}`` panel titles (e.g. class names for class-conditional
    maps of one channel).
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
        label = panel_labels[idx] if panel_labels is not None else f"ch {idx}"
        ax.set_title(label, fontsize=8)
    fig.suptitle(f"{title} (0 = always-negative, 1 = always-positive)")
    if image is not None:
        fig.colorbar(image, ax=axes, shrink=0.8, label="hard q")
    try:
        fig.savefig(path, dpi=110, bbox_inches="tight")
    finally:
        plt.close(fig)
    return path


def plot_decision_tree(
    node: Node,
    path: Path,
    leaf_label: Callable[[Leaf], str] | None = None,
    title: str = "Decompiled decision tree",
    max_terminals: int = 128,
) -> Path:
    """Render a decompiled tree as a node-and-edge diagram (phase 5).

    Branch nodes show the tested unit (``site[index]``); the left edge is the
    ``≤ 0`` (dead) side and the right edge the ``> 0`` (fires) side. Leaves
    show ``leaf_label(leaf)`` (default ``"affine"`` — pass e.g. a majority-
    class labeler); truncated regions render as ``…``. Refuses trees with more
    than ``max_terminals`` terminal nodes (the figure would be illegible).
    """
    stats = tree_stats(node)
    n_terminals = stats.n_leaves + stats.n_truncated
    if n_terminals > max_terminals:
        raise ValueError(
            f"tree has {n_terminals} terminal nodes; refusing to render more "
            f"than {max_terminals} (raise max_terminals to override)"
        )

    positions: dict[int, tuple[float, float]] = {}
    next_x = [0.0]

    def _place(current: Node, depth: int) -> float:
        if isinstance(current, Branch):
            x_low = _place(current.low, depth + 1)
            x_high = _place(current.high, depth + 1)
            x = (x_low + x_high) / 2
        else:
            x = next_x[0]
            next_x[0] += 1.0
        positions[id(current)] = (x, float(-depth))
        return x

    _place(node, 0)

    fig, ax = plt.subplots(
        figsize=(
            max(6.0, 0.55 * n_terminals),
            max(3.0, 1.0 * (stats.depth + 1)),
        )
    )
    ax.axis("off")

    def _draw(current: Node) -> None:
        x, y = positions[id(current)]
        if isinstance(current, Branch):
            for child, edge in ((current.low, "≤ 0"), (current.high, "> 0")):
                cx, cy = positions[id(child)]
                ax.plot([x, cx], [y, cy], color="grey", lw=0.8, zorder=1)
                ax.text(
                    (x + cx) / 2,
                    (y + cy) / 2,
                    edge,
                    fontsize=6,
                    color="grey",
                    ha="center",
                )
                _draw(child)
            text, color = f"{current.site}[{current.index}]", "lightyellow"
        elif isinstance(current, Leaf):
            text = leaf_label(current) if leaf_label is not None else "affine"
            color = "lightblue"
        else:
            text, color = "…", "mistyrose"
        ax.text(
            x,
            y,
            text,
            ha="center",
            va="center",
            fontsize=7,
            zorder=2,
            bbox={"boxstyle": "round", "fc": color, "ec": "grey"},
        )

    _draw(node)
    ax.set_title(title)
    try:
        fig.savefig(path, dpi=110, bbox_inches="tight")
    finally:
        plt.close(fig)
    return path


def plot_class_q_matrix(
    q: Tensor,
    path: Path,
    title: str = "Class-conditional firing rates",
    class_names: Sequence[str] | None = None,
) -> Path:
    """Heatmap of per-class hard q: rows are classes, columns are neurons.

    ``q`` is ``[num_classes, N]`` (e.g. from ``analysis.statistics.
    class_conditional_q``); a shared 0..1 scale makes class-selective units
    visible as vertical contrast.
    """
    if q.ndim != 2:
        raise ValueError(f"q must be [classes, neurons], got shape {tuple(q.shape)}")
    classes = int(q.shape[0])
    fig, ax = plt.subplots(figsize=(min(12.0, 2.0 + 0.25 * int(q.shape[1])), 3.5))
    image = ax.imshow(
        q.detach().cpu(), vmin=0.0, vmax=1.0, cmap="viridis", aspect="auto"
    )
    labels = (
        list(class_names)
        if class_names is not None
        else [str(c) for c in range(classes)]
    )
    ax.set_yticks(range(classes), labels)
    ax.set_xlabel("neuron")
    ax.set_ylabel("class")
    ax.set_title(title)
    fig.colorbar(image, ax=ax, shrink=0.9, label="hard q | class")
    fig.tight_layout()
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
