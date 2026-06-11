"""Phase-4 addendum: construct inputs that flip each ReLU's sign.

"Certified 0 neurons over the data box" does not by itself prove every ReLU is
flippable — the LP relaxation is sound, not exact, for layers ≥ 2. This script
upgrades the claim where possible by CONSTRUCTING witness inputs inside the
box:

- **Layer 1 (exact):** the box-LP optimum of an affine function is attained at
  a known corner — ``x⁺ = where(w > 0, hi, lo)`` maximizes ``w·x`` and the
  opposite corner minimizes it. Evaluating the model at those corners gives,
  per neuron, the true extreme pre-activations: if they straddle zero the
  neuron is **provably flippable with witnesses in hand**.
- **Layer 2 (lower bound):** projected sign-gradient ascent/descent inside the
  box searches for witnesses; successes are proofs, failures prove nothing.

Writes ``docs/research/assets/flip_witnesses_results.json``.

Usage:
    uv run --no-sync python scripts/flip_witnesses.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import torch

from kill_nonlinearities.analysis.ranges import data_box
from kill_nonlinearities.data.datasets import make_dataloaders
from kill_nonlinearities.experiments.run import config_from_json
from kill_nonlinearities.models.mlp import ReLUMLP
from kill_nonlinearities.training.checkpoint import load_checkpoint

ASSET_DIR = Path("docs/research/assets")
LAMBDAS = (0.0, 10.0)
PGD_STEPS = 300


def _final_checkpoint(run_dir: Path) -> Path:
    steps = {
        int(m.group(1)): p
        for p in run_dir.glob("step_*.pt")
        if (m := re.fullmatch(r"step_(\d+)\.pt", p.name))
    }
    return steps[max(steps)]


def _layer1_corner_extremes(
    model: ReLUMLP, lo: torch.Tensor, hi: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """True per-neuron layer-1 extremes, verified by forward passes at corners."""
    weight = model.linears[0].weight.detach()
    x_max = torch.where(weight > 0, hi, lo)  # row j maximizes neuron j
    x_min = torch.where(weight > 0, lo, hi)
    model.eval()
    with torch.no_grad():
        z_hi = model(x_max).pre_activations[0].diagonal()
        z_lo = model(x_min).pre_activations[0].diagonal()
    return z_lo, z_hi


def _layer2_pgd_extreme(
    model: ReLUMLP, lo: torch.Tensor, hi: torch.Tensor, sign: float
) -> torch.Tensor:
    """Best-found per-neuron layer-2 pre-activation extreme (sign=+1: max)."""
    n = model.linears[1].out_features
    x = ((lo + hi) / 2).repeat(n, 1).clone().requires_grad_(True)
    step = 0.05 * (hi - lo)
    best = torch.full((n,), -torch.inf, dtype=lo.dtype)
    for _ in range(PGD_STEPS):
        z = model(x).pre_activations[1].diagonal()
        best = torch.maximum(best, sign * z.detach())
        objective = (sign * z).sum()
        (grad,) = torch.autograd.grad(objective, x)
        with torch.no_grad():
            x += step * grad.sign()
            x.clamp_(lo, hi)
    with torch.no_grad():
        z = model(x).pre_activations[1].diagonal()
    return sign * torch.maximum(best, sign * z.detach())


def main() -> None:
    base = config_from_json(Path("configs/mnist.json"))
    train_loader, _, _ = make_dataloaders(base)
    box_lo, box_hi = data_box(train_loader)

    records: list[dict] = []
    for lam in LAMBDAS:
        model = ReLUMLP(base.model)
        load_checkpoint(_final_checkpoint(Path("runs") / f"{base.name}-{lam}"), model)
        model = model.double()
        model.eval()

        z_lo1, z_hi1 = _layer1_corner_extremes(model, box_lo, box_hi)
        flip1 = int(((z_hi1 > 0) & (z_lo1 <= 0)).sum())

        best_hi2 = _layer2_pgd_extreme(model, box_lo, box_hi, sign=+1.0)
        best_lo2 = _layer2_pgd_extreme(model, box_lo, box_hi, sign=-1.0)
        flip2 = int(((best_hi2 > 0) & (best_lo2 <= 0)).sum())

        records.append(
            {
                "lam": lam,
                "layer1_flippable_exact": flip1,
                "layer1_total": int(z_hi1.shape[0]),
                "layer2_flippable_witnessed": flip2,
                "layer2_total": int(best_hi2.shape[0]),
            }
        )
        print(
            f"λ={lam:<4g} layer1 flippable (exact corners): {flip1}/256   "
            f"layer2 flippable (PGD witnesses): {flip2}/256",
            flush=True,
        )

    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    (ASSET_DIR / "flip_witnesses_results.json").write_text(
        json.dumps({"runs": records}) + "\n"
    )


if __name__ == "__main__":
    main()
