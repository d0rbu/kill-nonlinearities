"""Offline lambda-sweep: accuracy / eliminable-neuron trade-off + (k, λ) surfaces.

Runs the phase-1a pipeline (``run_experiment``) on a base config across several
regularization strengths ``lambda`` with wandb disabled, then renders:

1. ``{prefix}.png``                    — headline trade-off (accuracy + eliminable counts vs λ)
2. ``{prefix}_acc_vs_k_by_lambda.png`` — accuracy-vs-k overlay, one curve per λ (val + test)
3. ``{prefix}_acc_surface.png``        — 3D surface: val accuracy as a function of (k, λ)
4. ``{prefix}_q_hist.png``             — per-λ histogram of hard q_i (dead vs always-on mass)

Per-λ records (full k-curves and the per-neuron q_i array included) accumulate in
``{prefix}_results.json`` after every run, so an interrupted sweep keeps its
finished runs and ``--plot-only`` re-renders all figures from the JSON without
retraining. Runs are keyed by λ and **merged** into an existing results JSON from
the same config (re-running a λ replaces its record), so a sweep can be extended
with just the new values, e.g. ``--lambdas 2 5 10``.

Usage:
    uv run --no-sync python scripts/lambda_sweep.py                  # MNIST defaults
    uv run --no-sync python scripts/lambda_sweep.py \
        --config configs/cifar10.json --prefix cifar10_lambda_sweep
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from kill_nonlinearities.experiments.run import (
    config_from_json,
    run_experiment,
)

LAMBDAS: tuple[float, ...] = (0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0)
DEFAULT_CONFIG = Path("configs/mnist.json")
DEFAULT_ASSET_DIR = Path("docs/research/assets")
LOW_ENTROPY_NATS = 0.05
DATASET_TITLES = {"mnist": "MNIST", "cifar10": "CIFAR-10"}


def run_one(base_config: Path, lam: float) -> dict[str, object]:
    """Run the pipeline at one lambda and return the sweep record."""
    base = config_from_json(base_config)
    reg = dataclasses.replace(base.reg, lam=lam)
    config = dataclasses.replace(base, reg=reg, name=f"{base.name}-{lam}")
    result = run_experiment(config)

    stats = result.neuron_stats
    k0 = result.k_points[0]  # k == 0 == the trained model (no surgery)
    dead = sum(1 for s in stats if s.q == 0.0)
    always_on = sum(1 for s in stats if s.q == 1.0)
    return {
        "lam": lam,
        "val_acc": k0.val_acc,
        "test_acc": k0.test_acc,
        "eliminable": dead + always_on,
        "dead": dead,
        "always_on": always_on,
        "low_entropy": sum(1 for s in stats if s.entropy < LOW_ENTROPY_NATS),
        "total": len(stats),
        "max_q": max(s.q for s in stats),
        # 6 decimals is lossless for q (a multiple of 1/|val|) and keeps the
        # JSON compact enough to commit for large (CNN) neuron counts.
        "q": [round(s.q, 6) for s in stats],
        "k": [p.k for p in result.k_points],
        "k_val": [p.val_acc for p in result.k_points],
        "k_test": [p.test_acc for p in result.k_points],
        "k_random_val": [p.val_acc for p in result.random_k_points],
        "k_random_test": [p.test_acc for p in result.random_k_points],
    }


def _save(fig: plt.Figure, path: Path) -> None:
    try:
        fig.savefig(path, dpi=110, bbox_inches="tight")
    finally:
        plt.close(fig)
    print(f"  wrote {path}")


def plot_tradeoff(runs: list[dict], dataset: str, path: Path) -> None:
    """Accuracy + eliminable/low-entropy neuron counts vs λ (ordinal λ axis)."""
    xs = list(range(len(runs)))
    labels = [f"{r['lam']:g}" for r in runs]
    fig, ax_acc = plt.subplots(figsize=(7.5, 5))
    ax_acc.plot(
        xs, [r["val_acc"] for r in runs], "o-", color="tab:blue", label="val accuracy"
    )
    ax_acc.plot(
        xs,
        [r["test_acc"] for r in runs],
        "s-",
        color="tab:orange",
        label="test accuracy",
    )
    ax_acc.set_xticks(xs, labels)
    ax_acc.set_xlabel("regularization strength λ")
    ax_acc.set_ylabel("accuracy (trained model, no surgery)")
    ax_acc.set_ylim(0.0, 1.0)

    ax_neur = ax_acc.twinx()
    ax_neur.plot(
        xs,
        [r["eliminable"] for r in runs],
        "^--",
        color="tab:green",
        label="eliminable (q∈{0,1})",
    )
    ax_neur.plot(
        xs,
        [r["low_entropy"] for r in runs],
        "v:",
        color="tab:red",
        label=f"H(q) < {LOW_ENTROPY_NATS} nats",
    )
    ax_neur.set_ylabel(f"# neurons (of {int(runs[0]['total'])})")

    lines = ax_acc.get_lines() + ax_neur.get_lines()
    ax_acc.legend(lines, [ln.get_label() for ln in lines], loc="center left")
    ax_acc.set_title(f"Accuracy vs. eliminable-neuron trade-off across λ ({dataset})")
    _save(fig, path)


def plot_acc_vs_k_by_lambda(runs: list[dict], dataset: str, path: Path) -> None:
    """Overlay the entropy-order acc-vs-k curve for every λ (val left, test right)."""
    fig, (ax_val, ax_test) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    cmap = plt.get_cmap("viridis")
    denom = max(len(runs) - 1, 1)
    for i, r in enumerate(runs):
        color = cmap(i / denom)
        label = f"λ={r['lam']:g}"
        ax_val.plot(
            r["k"], r["k_val"], marker="o", markersize=3, color=color, label=label
        )
        ax_val.axvline(int(r["eliminable"]), color=color, linestyle=":", alpha=0.5)
        ax_test.plot(
            r["k"], r["k_test"], marker="s", markersize=3, color=color, label=label
        )
    ax_val.set_title(f"val accuracy vs surgery k ({dataset})")
    ax_test.set_title(f"test accuracy vs surgery k ({dataset})")
    for ax in (ax_val, ax_test):
        ax.set_xlabel("k (neurons converted, entropy order)")
    ax_val.set_ylabel("accuracy")
    ax_val.legend(fontsize=8, title="dotted: lossless prefix", title_fontsize=8)
    _save(fig, path)


def plot_acc_surface(runs: list[dict], dataset: str, path: Path) -> None:
    """3D surface of val accuracy over (k, λ); λ spaced ordinally, labeled with values."""
    if len(runs) < 2:
        print(f"  skipping {path} (a surface needs >= 2 lambdas)")
        return
    k_arr = np.asarray(runs[0]["k"], dtype=float)
    x_grid, y_grid = np.meshgrid(k_arr, np.arange(len(runs), dtype=float))
    z_grid = np.asarray([r["k_val"] for r in runs], dtype=float)

    fig = plt.figure(figsize=(9, 6.5))
    ax = fig.add_subplot(projection="3d")
    surf = ax.plot_surface(
        x_grid, y_grid, z_grid, cmap="viridis", edgecolor="k", linewidth=0.2
    )
    ax.set_xlabel("k (neurons converted, entropy order)")
    ax.set_ylabel("λ")
    ax.set_yticks(range(len(runs)))
    ax.set_yticklabels([f"{r['lam']:g}" for r in runs])
    ax.set_zlabel("val accuracy")
    ax.zaxis.labelpad = 10
    ax.set_title(f"val accuracy vs (k, λ) — entropy-order masking ({dataset})")
    ax.view_init(elev=25, azim=-135)
    fig.colorbar(surf, shrink=0.55, pad=0.1, label="val accuracy")
    _save(fig, path)


def plot_q_hist(runs: list[dict], dataset: str, path: Path) -> None:
    """Histogram of hard q_i per λ: shows where the q∈{0,1} mass actually piles up."""
    fig, axes = plt.subplots(
        1, len(runs), figsize=(2.6 * len(runs), 3.0), sharey=True, squeeze=False
    )
    bins = [i / 40 for i in range(41)]
    for ax, r in zip(axes[0], runs, strict=True):
        ax.hist(r["q"], bins=bins, color="tab:blue")
        ax.axvline(0.5, color="grey", linestyle=":", linewidth=0.8)
        ax.set_yscale("log")
        ax.set_title(f"λ={r['lam']:g}", fontsize=10)
        ax.set_xlabel("hard $q_i$")
    axes[0][0].set_ylabel("# neurons (log)")
    fig.suptitle(f"Distribution of hard fraction-positive $q_i$ across λ ({dataset})")
    fig.tight_layout()
    _save(fig, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline λ-sweep for phase 1a.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--prefix", type=str, default="lambda_sweep")
    parser.add_argument("--lambdas", type=float, nargs="+", default=list(LAMBDAS))
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_ASSET_DIR)
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="re-render figures from the existing results JSON without retraining",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.out_dir / f"{args.prefix}_results.json"

    if args.plot_only:
        payload = json.loads(results_path.read_text())
    else:
        base = config_from_json(args.config)
        by_lam: dict[float, dict] = {}
        if results_path.exists():
            prev = json.loads(results_path.read_text())
            if prev.get("meta", {}).get("config") == str(args.config):
                by_lam = {float(r["lam"]): r for r in prev["runs"]}
                print(f"merging into {len(by_lam)} existing runs from {results_path}")
            else:
                print(
                    f"existing {results_path} is from a different config "
                    f"({prev.get('meta', {}).get('config')!r}); starting fresh"
                )
        payload: dict = {"meta": {}, "runs": []}
        for lam in args.lambdas:
            start = time.perf_counter()
            record = run_one(args.config, lam)
            by_lam[float(lam)] = record
            runs_sorted = sorted(by_lam.values(), key=lambda r: float(r["lam"]))
            payload = {
                "meta": {
                    "config": str(args.config),
                    "dataset": base.data.dataset,
                    "lambdas": [r["lam"] for r in runs_sorted],
                },
                "runs": runs_sorted,
            }
            # Compact dump: with per-position CNN runs the q arrays hold tens of
            # thousands of floats per lambda; indent=2 would 3x the file size.
            results_path.write_text(json.dumps(payload) + "\n")
            print(
                f"λ={lam:<5g} done in {time.perf_counter() - start:5.0f}s: "
                f"val={record['val_acc']:.4f} test={record['test_acc']:.4f} "
                f"dead={record['dead']:>3} always_on={record['always_on']:>3} "
                f"low_H={record['low_entropy']:>3}/{record['total']} "
                f"max_q={record['max_q']:.3f}",
                flush=True,
            )

    runs = payload["runs"]
    dataset = DATASET_TITLES.get(payload["meta"]["dataset"], payload["meta"]["dataset"])
    plot_tradeoff(runs, dataset, args.out_dir / f"{args.prefix}.png")
    plot_acc_vs_k_by_lambda(
        runs, dataset, args.out_dir / f"{args.prefix}_acc_vs_k_by_lambda.png"
    )
    plot_acc_surface(runs, dataset, args.out_dir / f"{args.prefix}_acc_surface.png")
    plot_q_hist(runs, dataset, args.out_dir / f"{args.prefix}_q_hist.png")

    print("lambda sweep complete:")
    for r in runs:
        print(
            f"  λ={r['lam']:<5g} val={r['val_acc']:.4f} test={r['test_acc']:.4f} "
            f"eliminable={int(r['eliminable']):>3} (dead={int(r['dead'])}, "
            f"on={int(r['always_on'])}) low_H={int(r['low_entropy']):>3}/{int(r['total'])}"
        )


if __name__ == "__main__":
    main()
