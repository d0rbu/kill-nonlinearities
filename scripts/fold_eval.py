"""Phase-2 fold evaluation over the trained λ-sweep checkpoints (MLPs).

For every ``runs/<name>-<lambda>/`` directory of the given base configs, load
the final checkpoint, classify neurons from val-split statistics at two
thresholds — ``exact`` (hard q exactly 0/1) and ``lowH`` (sign-entropy ≤ 0.05
nats, tie q>0.5 → IDENTITY) — apply the modes, then **fold + trim** the masked
model into a structurally smaller network and verify it against the masked
model over the FULL test set (max |Δlogits|, prediction agreement) while
recording accuracy and size reductions.

Writes ``docs/research/assets/fold_eval_results.json`` and ``fold_eval.png``.

Usage:
    uv run --no-sync python scripts/fold_eval.py
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from kill_nonlinearities.analysis.selection import assign_modes
from kill_nonlinearities.analysis.statistics import (
    collect_pre_activations,
    neuron_stats,
)
from kill_nonlinearities.data.datasets import make_dataloaders
from kill_nonlinearities.experiments.run import config_from_json
from kill_nonlinearities.models.mlp import ReLUMLP
from kill_nonlinearities.surgery.apply import apply_modes
from kill_nonlinearities.surgery.fold import fold_mlp, folded_stats, trim_folded
from kill_nonlinearities.training.checkpoint import load_checkpoint

ASSET_DIR = Path("docs/research/assets")
RUNS_DIR = Path("runs")
LOW_ENTROPY_NATS = 0.05
CONFIGS = ("configs/mnist.json", "configs/cifar10.json", "configs/cifar10-wide.json")


def _final_checkpoint(run_dir: Path) -> Path | None:
    steps = {
        int(m.group(1)): p
        for p in run_dir.glob("step_*.pt")
        if (m := re.fullmatch(r"step_(\d+)\.pt", p.name))
    }
    return steps[max(steps)] if steps else None


def _eval_pair(
    masked: ReLUMLP, folded: torch.nn.Module, loader, device: str
) -> dict[str, float]:
    """Accuracies, prediction agreement, and max |Δlogits| over ``loader``."""
    masked.eval()
    folded.eval()
    correct_m = correct_f = agree = total = 0
    max_diff = 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            lm = masked(x).logits
            lf = folded(x)
            max_diff = max(max_diff, float((lm - lf).abs().max()))
            pm, pf = lm.argmax(dim=1), lf.argmax(dim=1)
            correct_m += int((pm == y).sum())
            correct_f += int((pf == y).sum())
            agree += int((pm == pf).sum())
            total += int(y.shape[0])
    return {
        "masked_acc": correct_m / total,
        "folded_acc": correct_f / total,
        "agreement": agree / total,
        "max_logit_diff": max_diff,
    }


def main() -> None:
    records: list[dict[str, object]] = []
    for config_path in CONFIGS:
        base = config_from_json(Path(config_path))
        _, val_loader, test_loader = make_dataloaders(base)
        for run_dir in sorted(RUNS_DIR.glob(f"{base.name}-*")):
            lam = float(run_dir.name.removeprefix(f"{base.name}-"))
            ckpt = _final_checkpoint(run_dir)
            if ckpt is None:
                print(f"  {run_dir.name}: no checkpoint, skipping")
                continue
            model = ReLUMLP(base.model)
            load_checkpoint(ckpt, model)
            model.eval()

            stats = neuron_stats(
                collect_pre_activations(model, val_loader, base.train.device)
            )
            widths = {
                site: int(act.mode.shape[0])
                for site, act in zip(model.site_names, model.activations, strict=True)
            }
            original_params = sum(p.numel() for p in model.parameters())

            for threshold, selection in (
                ("exact", [s for s in stats if s.q in (0.0, 1.0)]),
                ("lowH", [s for s in stats if s.entropy <= LOW_ENTROPY_NATS]),
            ):
                masked = copy.deepcopy(model)
                apply_modes(
                    masked,
                    assign_modes(selection, tie_break="identity", widths=widths),
                )
                folded = trim_folded(fold_mlp(masked))
                fstats = folded_stats(folded)
                metrics = _eval_pair(masked, folded, test_loader, base.train.device)
                record: dict[str, object] = {
                    "dataset": base.name,
                    "lam": lam,
                    "threshold": threshold,
                    "converted": len(selection),
                    "total_neurons": len(stats),
                    "nonlinear_widths": list(fstats.nonlinear_widths),
                    "carry_widths": list(fstats.carry_widths),
                    "original_params": original_params,
                    "folded_params": fstats.params,
                    **metrics,
                }
                records.append(record)
                widths_str = "x".join(str(w) for w in fstats.nonlinear_widths)
                print(
                    f"{base.name:<14} λ={lam:<5g} {threshold:<5} "
                    f"relu={widths_str:>9} params {original_params}→{fstats.params} "
                    f"acc m={metrics['masked_acc']:.4f} f={metrics['folded_acc']:.4f} "
                    f"agree={metrics['agreement']:.4f} "
                    f"maxΔ={metrics['max_logit_diff']:.2e}",
                    flush=True,
                )

    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    (ASSET_DIR / "fold_eval_results.json").write_text(
        json.dumps({"runs": records}) + "\n"
    )
    _plot(records, ASSET_DIR / "fold_eval.png")


def _plot(records: list[dict], path: Path) -> None:
    """Remaining ReLU units + folded accuracy vs λ, one panel per dataset."""
    datasets = sorted({r["dataset"] for r in records})
    fig, axes = plt.subplots(
        1, len(datasets), figsize=(6 * len(datasets), 4.5), squeeze=False
    )
    for ax, dataset in zip(axes[0], datasets, strict=True):
        rows = [r for r in records if r["dataset"] == dataset]
        lams = sorted({r["lam"] for r in rows})
        xs = list(range(len(lams)))
        ax2 = ax.twinx()
        for threshold, color in (("exact", "tab:blue"), ("lowH", "tab:red")):
            sub = {r["lam"]: r for r in rows if r["threshold"] == threshold}
            ax.plot(
                xs,
                [sum(sub[lam]["nonlinear_widths"]) for lam in lams],
                "o-",
                color=color,
                label=f"ReLUs left ({threshold})",
            )
            ax2.plot(
                xs,
                [sub[lam]["folded_acc"] for lam in lams],
                "s--",
                color=color,
                alpha=0.6,
                label=f"folded acc ({threshold})",
            )
        ax.set_xticks(xs, [f"{lam:g}" for lam in lams])
        ax.set_xlabel("λ")
        ax.set_ylabel("remaining nonlinear units")
        ax2.set_ylabel("folded test accuracy")
        ax2.set_ylim(0.0, 1.0)
        ax.set_title(dataset)
        lines = ax.get_lines() + ax2.get_lines()
        ax.legend(lines, [ln.get_label() for ln in lines], fontsize=8)
    fig.tight_layout()
    try:
        fig.savefig(path, dpi=110, bbox_inches="tight")
    finally:
        plt.close(fig)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
