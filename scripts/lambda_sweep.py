"""Offline lambda-sweep: accuracy / eliminable-neuron trade-off curve.

Runs the phase-1a pipeline (``run_experiment``) on MNIST across several
regularization strengths ``lambda`` with wandb disabled, then plots the headline
trade-off: how baseline accuracy and the number of eliminable (sign-consistent)
neurons move as ``lambda`` increases.

Usage:
    uv run --no-sync python scripts/lambda_sweep.py

Writes ``docs/research/assets/lambda_sweep.png`` and ``..._results.json``.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from kill_nonlinearities.experiments.run import (
    config_from_json,
    run_experiment,
)

LAMBDAS: tuple[float, ...] = (0.0, 0.01, 0.05, 0.1, 0.2)
BASE_CONFIG = Path("configs/mnist.json")
ASSET_DIR = Path("docs/research/assets")


def run_one(lam: float) -> dict[str, float]:
    """Run MNIST at one lambda and return the trade-off metrics."""
    base = config_from_json(BASE_CONFIG)
    reg = dataclasses.replace(base.reg, lam=lam)
    config = dataclasses.replace(base, reg=reg, name=f"mnist-lambda-{lam}")
    result = run_experiment(config)

    k0 = result.k_points[0]  # k == 0 == the trained model (no surgery)
    eliminable = sum(1 for s in result.neuron_stats if s.q in (0.0, 1.0))
    low_entropy = sum(1 for s in result.neuron_stats if s.entropy < 0.05)
    total = len(result.neuron_stats)
    return {
        "lam": lam,
        "val_acc": k0.val_acc,
        "test_acc": k0.test_acc,
        "eliminable": float(eliminable),
        "low_entropy": float(low_entropy),
        "total": float(total),
    }


def main() -> None:
    results = [run_one(lam) for lam in LAMBDAS]
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    (ASSET_DIR / "lambda_sweep_results.json").write_text(json.dumps(results, indent=2))

    lams = [r["lam"] for r in results]
    fig, ax_acc = plt.subplots(figsize=(7, 5))
    ax_acc.plot(
        lams,
        [r["val_acc"] for r in results],
        "o-",
        color="tab:blue",
        label="val accuracy",
    )
    ax_acc.plot(
        lams,
        [r["test_acc"] for r in results],
        "s-",
        color="tab:orange",
        label="test accuracy",
    )
    ax_acc.set_xlabel("regularization strength λ")
    ax_acc.set_ylabel("accuracy (trained model, no surgery)")
    ax_acc.set_ylim(0.0, 1.0)

    ax_neur = ax_acc.twinx()
    ax_neur.plot(
        lams,
        [r["eliminable"] for r in results],
        "^--",
        color="tab:green",
        label="eliminable (q∈{0,1})",
    )
    ax_neur.plot(
        lams,
        [r["low_entropy"] for r in results],
        "v:",
        color="tab:red",
        label="H(q) < 0.05 nats",
    )
    ax_neur.set_ylabel(f"# neurons (of {int(results[0]['total'])})")

    lines = ax_acc.get_lines() + ax_neur.get_lines()
    ax_acc.legend(lines, [ln.get_label() for ln in lines], loc="center left")
    ax_acc.set_title("Accuracy vs. eliminable-neuron trade-off across λ (MNIST)")
    try:
        fig.savefig(ASSET_DIR / "lambda_sweep.png", dpi=110, bbox_inches="tight")
    finally:
        plt.close(fig)

    print("lambda sweep complete:")
    for r in results:
        print(
            f"  λ={r['lam']:<5} val={r['val_acc']:.4f} test={r['test_acc']:.4f} "
            f"eliminable={int(r['eliminable']):>3} low_H={int(r['low_entropy']):>3}/{int(r['total'])}"
        )


if __name__ == "__main__":
    main()
