"""Phase-4 supplement: certified consistency vs input-box scale.

The full data box certifies nothing (see ``range_certify.py``): its hull
contains wildly off-manifold points. This sweep shrinks the box toward its
center by a factor ``s`` and asks at which scale certification appears, and
whether the regularizer buys certified margin — comparing MNIST λ=0 vs λ=10
with the LP bounds.

Writes ``docs/research/assets/range_scale_results.json`` and
``range_scale.png``.

Usage:
    uv run --no-sync python scripts/range_scale_sweep.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from kill_nonlinearities.analysis.ranges import certified_modes, data_box, lp_ranges
from kill_nonlinearities.data.datasets import make_dataloaders
from kill_nonlinearities.experiments.run import config_from_json
from kill_nonlinearities.models.activations import ActivationMode
from kill_nonlinearities.models.mlp import ReLUMLP
from kill_nonlinearities.training.checkpoint import load_checkpoint

ASSET_DIR = Path("docs/research/assets")
LAMBDAS = (0.0, 10.0)
SCALES = (0.01, 0.05, 0.1, 0.25, 0.5, 1.0)


def _final_checkpoint(run_dir: Path) -> Path:
    steps = {
        int(m.group(1)): p
        for p in run_dir.glob("step_*.pt")
        if (m := re.fullmatch(r"step_(\d+)\.pt", p.name))
    }
    return steps[max(steps)]


def main() -> None:
    base = config_from_json(Path("configs/mnist.json"))
    train_loader, _, _ = make_dataloaders(base)
    box_lo, box_hi = data_box(train_loader)
    center = (box_lo + box_hi) / 2
    half = (box_hi - box_lo) / 2

    records: list[dict] = []
    for lam in LAMBDAS:
        model = ReLUMLP(base.model)
        load_checkpoint(_final_checkpoint(Path("runs") / f"{base.name}-{lam}"), model)
        model.eval()
        for scale in SCALES:
            lo = center - scale * half
            hi = center + scale * half
            modes = certified_modes(lp_ranges(model, lo, hi))
            dead = sum(
                int((m == int(ActivationMode.ZERO)).sum()) for m in modes.values()
            )
            on = sum(
                int((m == int(ActivationMode.IDENTITY)).sum()) for m in modes.values()
            )
            records.append({"lam": lam, "scale": scale, "dead": dead, "on": on})
            print(
                f"λ={lam:<4g} scale={scale:<5g} certified dead={dead:>3} on={on:>3}",
                flush=True,
            )

    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    (ASSET_DIR / "range_scale_results.json").write_text(
        json.dumps({"runs": records}) + "\n"
    )

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for lam, color in zip(LAMBDAS, ("tab:blue", "tab:red"), strict=True):
        rows = [r for r in records if r["lam"] == lam]
        ax.plot(
            [r["scale"] for r in rows],
            [r["dead"] + r["on"] for r in rows],
            "o-",
            color=color,
            label=f"λ={lam:g}",
        )
    ax.set_xscale("log")
    ax.set_xlabel("input-box scale s (fraction of the data box, around its center)")
    ax.set_ylabel("# certified sign-consistent neurons (LP, of 512)")
    ax.set_title("MNIST: certification appears as the box shrinks toward the data")
    ax.legend()
    fig.tight_layout()
    try:
        fig.savefig(ASSET_DIR / "range_scale.png", dpi=110, bbox_inches="tight")
    finally:
        plt.close(fig)
    print(f"wrote {ASSET_DIR / 'range_scale.png'}")


if __name__ == "__main__":
    main()
