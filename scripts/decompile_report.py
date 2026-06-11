"""Phase-5 report: decompile networks into conditionals; tree size vs λ.

Two experiments:

1. **2D toy** — train a tiny 2→8→2 MLP on a ring-vs-blob task at λ ∈ {0, 0.3}
   (inline training loop), decompile each over the data box, render the full
   nested-conditional program (committed as ``.txt``) and a decision-region
   figure. The regularized program should be visibly shorter.
2. **MNIST** — decompile the trained λ-sweep checkpoints (λ ∈ {0, 1, 10}) over
   centered sub-boxes of the data box (scales 0.25 / 0.5, where phase-4 showed
   certification is meaningful), with a leaf budget; record branch/leaf/
   truncation counts and exactness checks of the tree against the network on
   sampled in-box inputs.

Writes ``docs/research/assets/decompile_*`` artifacts.

Usage:
    uv run --no-sync python scripts/decompile_report.py
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from collections import Counter

import matplotlib.pyplot as plt
import torch
from torch import nn

from kill_nonlinearities.analysis.decompile import (
    Branch,
    Leaf,
    Node,
    decompile_mlp,
    evaluate_tree,
    render_tree,
    route_leaves,
    tree_stats,
)
from kill_nonlinearities.analysis.ranges import data_box
from kill_nonlinearities.config import ModelConfig
from kill_nonlinearities.data.datasets import make_dataloaders
from kill_nonlinearities.experiments.run import config_from_json
from kill_nonlinearities.models.mlp import ReLUMLP
from kill_nonlinearities.regularization.loss import sign_consistency_loss
from kill_nonlinearities.training.checkpoint import load_checkpoint
from kill_nonlinearities.training.schedule import TemperatureSchedule
from kill_nonlinearities.viz.network import plot_decision_tree

ASSET_DIR = Path("docs/research/assets")
TOY_LAMBDAS = (0.0, 0.3)
TOY_STEPS = 800
MNIST_LAMBDAS = (0.0, 1.0, 10.0)
# s=0.5 was explored once (λ=0): budget-dominated (228/256 truncated, depth
# 237) at ~45 min/tree — dropped from the committed sweep for cost.
MNIST_SCALES = (0.25,)
MAX_LEAVES = 256


def _toy_data(n: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    x = torch.rand(n, 2, generator=gen) * 4.0 - 2.0  # the box [-2, 2]^2
    y = (x.norm(dim=1) < 1.0).long()  # blob inside the unit circle vs outside
    return x, y


def toy_experiment() -> list[dict]:
    records: list[dict] = []
    x, y = _toy_data(4096, seed=0)
    schedule = TemperatureSchedule(
        kind="exponential", tau_start=1.0, tau_end=0.1, total_steps=TOY_STEPS
    )
    for lam in TOY_LAMBDAS:
        torch.manual_seed(0)
        model = ReLUMLP(ModelConfig(input_dim=2, hidden_dims=(8,), output_dim=2))
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
        criterion = nn.CrossEntropyLoss()
        model.train()
        for step in range(TOY_STEPS):
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out.logits, y) + lam * sign_consistency_loss(
                out.pre_activations, schedule(step), 1e-6
            )
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            accuracy = float((model(x).logits.argmax(dim=1) == y).float().mean())

        lo = torch.full((2,), -2.0, dtype=torch.float64)
        hi = torch.full((2,), 2.0, dtype=torch.float64)
        tree = decompile_mlp(model.double(), lo, hi, max_leaves=4096)
        stats = tree_stats(tree)
        program = render_tree(tree, max_terms=2)
        (ASSET_DIR / f"decompile_toy_lam{lam:g}.txt").write_text(program + "\n")

        # Exactness of the program on a fresh in-box sample.
        probe, _ = _toy_data(2048, seed=1)
        diff = float(
            (evaluate_tree(tree, probe.double()) - model(probe.double()).logits)
            .abs()
            .max()
        )
        records.append(
            {
                "experiment": "toy",
                "lam": lam,
                "accuracy": accuracy,
                "branches": stats.n_branches,
                "leaves": stats.n_leaves,
                "truncated": stats.n_truncated,
                "depth": stats.depth,
                "max_diff": diff,
            }
        )
        print(
            f"toy λ={lam:g}: acc={accuracy:.3f} branches={stats.n_branches} "
            f"leaves={stats.n_leaves} depth={stats.depth} maxΔ={diff:.2e}",
            flush=True,
        )
        _plot_toy_regions(model, tree, lam, stats.n_leaves)
        _plot_toy_tree(tree, x.double(), y, lam)
    return records


def _plot_toy_tree(tree: Node, x: torch.Tensor, y: torch.Tensor, lam: float) -> None:
    """Tree diagram with leaves labeled by their majority class + sample count."""
    counts: dict[int, Counter] = {}
    for leaf, label in zip(route_leaves(tree, x), y.tolist(), strict=True):
        counts.setdefault(id(leaf), Counter())[label] += 1

    def leaf_label(leaf: Leaf) -> str:
        tally = counts.get(id(leaf))
        if tally is None:
            return "no data"
        cls, n = tally.most_common(1)[0]
        return f"{'in' if cls == 1 else 'out'}\n({n})"

    plot_decision_tree(
        tree,
        ASSET_DIR / f"decompile_toy_tree_lam{lam:g}.png",
        leaf_label=leaf_label,
        title=f"toy λ={lam:g}: decompiled decision tree "
        "(leaf = majority class, training samples)",
    )


Point = tuple[float, float]


def _halfplane_value(point: Point, w: list[float], b: float) -> float:
    return w[0] * point[0] + w[1] * point[1] + b


def _clip_polygon(
    poly: list[Point], w: list[float], b: float, keep_high: bool
) -> list[Point]:
    """Sutherland-Hodgman clip of a convex polygon against one half-plane."""
    out: list[Point] = []
    for i in range(len(poly)):
        cur, nxt = poly[i], poly[(i + 1) % len(poly)]
        cv, nv = _halfplane_value(cur, w, b), _halfplane_value(nxt, w, b)
        cur_in = cv > 0 if keep_high else cv <= 0
        nxt_in = nv > 0 if keep_high else nv <= 0
        if cur_in:
            out.append(cur)
        if cur_in != nxt_in:
            t = cv / (cv - nv)
            out.append((cur[0] + t * (nxt[0] - cur[0]), cur[1] + t * (nxt[1] - cur[1])))
    return out


def _chord(poly: list[Point], w: list[float], b: float) -> list[Point]:
    """The segment of the line w·v + b = 0 inside a convex polygon."""
    points: list[Point] = []
    for i in range(len(poly)):
        cur, nxt = poly[i], poly[(i + 1) % len(poly)]
        cv, nv = _halfplane_value(cur, w, b), _halfplane_value(nxt, w, b)
        if (cv > 0) != (nv > 0):
            t = cv / (cv - nv)
            points.append(
                (cur[0] + t * (nxt[0] - cur[0]), cur[1] + t * (nxt[1] - cur[1]))
            )
    return points[:2]


def _draw_boundaries(ax, node, poly: list[Point]) -> None:
    """Draw EVERY branch hyperplane clipped to its own region (recursively).

    This renders all region borders — including ones separating two regions
    whose leaves predict the same class.
    """
    if not isinstance(node, Branch) or len(poly) < 3:
        return
    w = [float(node.weight[0]), float(node.weight[1])]
    segment = _chord(poly, w, node.bias)
    if len(segment) == 2:
        ax.plot(
            [segment[0][0], segment[1][0]],
            [segment[0][1], segment[1][1]],
            color="black",
            lw=0.7,
        )
    _draw_boundaries(ax, node.low, _clip_polygon(poly, w, node.bias, keep_high=False))
    _draw_boundaries(ax, node.high, _clip_polygon(poly, w, node.bias, keep_high=True))


def _plot_toy_regions(model: ReLUMLP, tree, lam: float, leaves: int) -> None:
    grid = torch.linspace(-2.0, 2.0, 301, dtype=torch.float64)
    xx, yy = torch.meshgrid(grid, grid, indexing="xy")
    points = torch.stack([xx.flatten(), yy.flatten()], dim=1)
    labels = evaluate_tree(tree, points).argmax(dim=1).reshape(xx.shape)
    fig, ax = plt.subplots(figsize=(4.5, 4.2))
    ax.imshow(
        labels,
        extent=(-2, 2, -2, 2),
        origin="lower",
        cmap="coolwarm",
        alpha=0.6,
    )
    # Every region border, even between same-class regions.
    _draw_boundaries(ax, tree, [(-2.0, -2.0), (2.0, -2.0), (2.0, 2.0), (-2.0, 2.0)])
    circle = plt.Circle((0, 0), 1.0, fill=False, color="black", linestyle=":")
    ax.add_patch(circle)
    ax.set_title(
        f"toy λ={lam:g}: decompiled program ({leaves} leaves; all region borders)"
    )
    fig.tight_layout()
    try:
        fig.savefig(ASSET_DIR / f"decompile_toy_lam{lam:g}.png", dpi=110)
    finally:
        plt.close(fig)


def _final_checkpoint(run_dir: Path) -> Path:
    steps = {
        int(m.group(1)): p
        for p in run_dir.glob("step_*.pt")
        if (m := re.fullmatch(r"step_(\d+)\.pt", p.name))
    }
    return steps[max(steps)]


def mnist_experiment() -> list[dict]:
    base = config_from_json(Path("configs/mnist.json"))
    train_loader, _, _ = make_dataloaders(base)
    box_lo, box_hi = data_box(train_loader)
    center = (box_lo + box_hi) / 2
    half = (box_hi - box_lo) / 2
    records: list[dict] = []
    for lam in MNIST_LAMBDAS:
        model = ReLUMLP(base.model)
        load_checkpoint(_final_checkpoint(Path("runs") / f"{base.name}-{lam}"), model)
        model = model.double()
        model.eval()
        for scale in MNIST_SCALES:
            lo = center - scale * half
            hi = center + scale * half
            t0 = time.perf_counter()
            tree = decompile_mlp(model, lo, hi, max_leaves=MAX_LEAVES)
            seconds = time.perf_counter() - t0
            stats = tree_stats(tree)

            gen = torch.Generator().manual_seed(7)
            probe = lo + torch.rand(256, 784, generator=gen, dtype=torch.float64) * (
                hi - lo
            )
            with torch.no_grad():
                expected = model(probe).logits
            if stats.n_truncated == 0:
                diff = float((evaluate_tree(tree, probe) - expected).abs().max())
            else:
                diff = float("nan")  # probes may land in truncated regions
            records.append(
                {
                    "experiment": "mnist",
                    "lam": lam,
                    "scale": scale,
                    "branches": stats.n_branches,
                    "leaves": stats.n_leaves,
                    "truncated": stats.n_truncated,
                    "depth": stats.depth,
                    "max_diff": diff,
                    "seconds": seconds,
                }
            )
            print(
                f"mnist λ={lam:<4g} scale={scale:g}: branches={stats.n_branches:>4} "
                f"leaves={stats.n_leaves:>4} truncated={stats.n_truncated:>3} "
                f"depth={stats.depth:>3} maxΔ={diff:.2e} ({seconds:.1f}s)",
                flush=True,
            )
    return records


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Phase-5 decompilation report.")
    parser.add_argument(
        "--toy-only",
        action="store_true",
        help="regenerate only the (fast) toy artifacts, skipping the MNIST trees",
    )
    args = parser.parse_args()

    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    records = toy_experiment()
    if not args.toy_only:
        records += mnist_experiment()
        (ASSET_DIR / "decompile_results.json").write_text(
            json.dumps({"runs": records}) + "\n"
        )
    print("wrote", ASSET_DIR / "decompile_results.json")


if __name__ == "__main__":
    main()
