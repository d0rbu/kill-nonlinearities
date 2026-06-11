"""Phase-4 certification: range analysis over the trained λ-sweep checkpoints.

For each checkpoint, build the input box as the per-pixel min/max of the train
split, compute pre-activation ranges with IBP and with the LP (HiGHS) method,
and count **certified** sign-consistent neurons — neurons provably dead /
always-on for EVERY input in the box, a strictly stronger statement than the
empirical hard q. Cross-validates certified neurons against the empirical
exact-q sets (certified ⇒ empirical must hold; violations are reported loudly)
and records IBP-vs-LP tightness.

Writes ``docs/research/assets/range_certify_results.json`` and
``range_certify.png``.

Usage:
    uv run --no-sync python scripts/range_certify.py
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from kill_nonlinearities.analysis.ranges import (
    certified_modes,
    data_box,
    interval_ranges,
    lp_ranges,
)
from kill_nonlinearities.analysis.statistics import (
    collect_pre_activations,
    neuron_stats,
)
from kill_nonlinearities.data.datasets import make_dataloaders
from kill_nonlinearities.experiments.run import config_from_json
from kill_nonlinearities.models.activations import ActivationMode
from kill_nonlinearities.models.mlp import ReLUMLP
from kill_nonlinearities.training.checkpoint import load_checkpoint

ASSET_DIR = Path("docs/research/assets")
TARGETS = (
    ("configs/mnist.json", (0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0)),
    ("configs/cifar10.json", (0.0, 1.0, 10.0)),
)


def _final_checkpoint(run_dir: Path) -> Path:
    steps = {
        int(m.group(1)): p
        for p in run_dir.glob("step_*.pt")
        if (m := re.fullmatch(r"step_(\d+)\.pt", p.name))
    }
    return steps[max(steps)]


def _counts(modes: dict[str, torch.Tensor]) -> tuple[int, int]:
    dead = sum(int((m == int(ActivationMode.ZERO)).sum()) for m in modes.values())
    on = sum(int((m == int(ActivationMode.IDENTITY)).sum()) for m in modes.values())
    return dead, on


def main() -> None:
    records: list[dict] = []
    for config_path, lambdas in TARGETS:
        base = config_from_json(Path(config_path))
        train_loader, val_loader, _ = make_dataloaders(base)
        box_lo, box_hi = data_box(train_loader)
        for lam in lambdas:
            model = ReLUMLP(base.model)
            load_checkpoint(
                _final_checkpoint(Path("runs") / f"{base.name}-{lam}"), model
            )
            model.eval()

            t0 = time.perf_counter()
            ibp = interval_ranges(model, box_lo, box_hi)
            ibp_seconds = time.perf_counter() - t0
            t0 = time.perf_counter()
            lp = lp_ranges(model, box_lo, box_hi)
            lp_seconds = time.perf_counter() - t0

            ibp_modes = certified_modes(ibp)
            lp_modes = certified_modes(lp)
            stats = neuron_stats(
                collect_pre_activations(model, val_loader, base.train.device)
            )
            emp_dead = sum(1 for s in stats if s.q == 0.0)
            emp_on = sum(1 for s in stats if s.q == 1.0)

            # Cross-validation: certified (whole box) must imply empirical (val).
            violations = 0
            by_site: dict[str, list] = {}
            for s in stats:
                by_site.setdefault(s.site, []).append(s)
            for site, site_stats in by_site.items():
                mode = lp_modes[site]
                for s in site_stats:
                    code = int(mode[s.index])
                    if code == int(ActivationMode.ZERO) and s.q != 0.0:
                        violations += 1
                    if code == int(ActivationMode.IDENTITY) and s.q != 1.0:
                        violations += 1
            if violations:
                print(f"!! SOUNDNESS VIOLATIONS: {violations} (investigate)")

            width_ratio = float(
                torch.cat([r.pre_upper - r.pre_lower for r in lp])
                .div(
                    torch.cat([r.pre_upper - r.pre_lower for r in ibp]).clamp(min=1e-12)
                )
                .mean()
            )
            ibp_dead, ibp_on = _counts(ibp_modes)
            lp_dead, lp_on = _counts(lp_modes)
            record = {
                "dataset": base.name,
                "lam": lam,
                "total": len(stats),
                "ibp_dead": ibp_dead,
                "ibp_on": ibp_on,
                "lp_dead": lp_dead,
                "lp_on": lp_on,
                "empirical_dead": emp_dead,
                "empirical_on": emp_on,
                "violations": violations,
                "lp_width_over_ibp": width_ratio,
                "ibp_seconds": ibp_seconds,
                "lp_seconds": lp_seconds,
            }
            records.append(record)
            print(
                f"{base.name:<14} λ={lam:<5g} certified dead/on: "
                f"IBP {ibp_dead}/{ibp_on}  LP {lp_dead}/{lp_on}  "
                f"empirical {emp_dead}/{emp_on}  "
                f"LP width ratio {width_ratio:.3f}  "
                f"({ibp_seconds:.1f}s / {lp_seconds:.1f}s)",
                flush=True,
            )

    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    (ASSET_DIR / "range_certify_results.json").write_text(
        json.dumps({"runs": records}) + "\n"
    )
    _plot(records, ASSET_DIR / "range_certify.png")


def _plot(records: list[dict], path: Path) -> None:
    datasets = sorted({r["dataset"] for r in records})
    fig, axes = plt.subplots(
        1, len(datasets), figsize=(6 * len(datasets), 4.5), squeeze=False
    )
    for ax, dataset in zip(axes[0], datasets, strict=True):
        rows = sorted(
            (r for r in records if r["dataset"] == dataset), key=lambda r: r["lam"]
        )
        xs = list(range(len(rows)))
        for key, style, label in (
            ("ibp", "^:", "certified (IBP)"),
            ("lp", "o-", "certified (LP)"),
            ("empirical", "s--", "empirical exact (val)"),
        ):
            ax.plot(
                xs,
                [r[f"{key}_dead"] + r[f"{key}_on"] for r in rows],
                style,
                label=label,
            )
        ax.set_xticks(xs, [f"{r['lam']:g}" for r in rows])
        ax.set_xlabel("λ")
        ax.set_ylabel("# sign-consistent neurons")
        ax.set_title(f"{dataset}: certified (whole input box) vs empirical")
        ax.legend(fontsize=8)
    fig.tight_layout()
    try:
        fig.savefig(path, dpi=110, bbox_inches="tight")
    finally:
        plt.close(fig)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
