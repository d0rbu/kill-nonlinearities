"""Phase-3 gallery: visualize the de-nonlinearized networks from on-disk artifacts.

Renders, into ``docs/research/assets/denonlin/``:

- **MNIST** (from the trained λ-sweep checkpoints, lowH modes, folded+trimmed):
  the per-site mode composition, the input-space filters of the SURVIVING
  nonlinear units (what the remaining ReLUs look at), and — for the fully
  folded λ=10 run — the per-class templates of the resulting bare affine map.
- **CIFAR-10 CNN** (from the committed per-position sweep results JSON, no
  retraining): per-channel spatial maps of hard q for each conv site at
  λ ∈ {0, 1, 10} — where the nonlinearity lives spatially.

Usage:
    uv run --no-sync python scripts/denonlin_gallery.py
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path

import torch

from kill_nonlinearities.analysis.selection import assign_modes
from kill_nonlinearities.analysis.statistics import (
    class_conditional_q,
    collect_pre_activations,
    neuron_stats,
)
from kill_nonlinearities.data.datasets import make_dataloaders
from kill_nonlinearities.experiments.run import config_from_json
from kill_nonlinearities.models import build_model
from kill_nonlinearities.models.activations import ActivationMode
from kill_nonlinearities.models.mlp import ReLUMLP
from kill_nonlinearities.surgery.apply import apply_modes
from kill_nonlinearities.surgery.fold import fold_mlp, folded_stats, trim_folded
from kill_nonlinearities.training.checkpoint import load_checkpoint
from kill_nonlinearities.viz.network import (
    plot_class_q_matrix,
    plot_folded_dag,
    plot_input_filters,
    plot_mode_composition,
    plot_spatial_q_map,
)

OUT_DIR = Path("docs/research/assets/denonlin")
LOW_ENTROPY_NATS = 0.05
MNIST_LAMBDAS = (0.5, 1.0, 10.0)
CNN_LAMBDAS = (0.0, 1.0, 10.0)
CNN_SITES = (("conv0", 32, 32), ("conv1", 64, 16), ("conv2", 128, 8))
CNN_RESULTS = Path("docs/research/assets/cifar10_cnn_sweep_results.json")
CIFAR_CLASSES = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)
# Channels picked from the lam=10 conv0 q-map: 14 reads ~0.5 everywhere
# ("switching"), 8 is solid dead, 0 solid always-on.
CNN_CLASS_CHANNELS = (14, 8, 0)


def _final_checkpoint(run_dir: Path) -> Path:
    steps = {
        int(m.group(1)): p
        for p in run_dir.glob("step_*.pt")
        if (m := re.fullmatch(r"step_(\d+)\.pt", p.name))
    }
    return steps[max(steps)]


def mnist_gallery() -> None:
    base = config_from_json(Path("configs/mnist.json"))
    _, val_loader, _ = make_dataloaders(base)
    for lam in MNIST_LAMBDAS:
        model = ReLUMLP(base.model)
        # Sweep run dirs are named with str(float): "mnist-lambda-1.0", not "-1".
        load_checkpoint(_final_checkpoint(Path("runs") / f"{base.name}-{lam}"), model)
        model.eval()
        stats = neuron_stats(
            collect_pre_activations(model, val_loader, base.train.device)
        )
        plot_mode_composition(
            stats,
            LOW_ENTROPY_NATS,
            OUT_DIR / f"mnist_composition_lam{lam:g}.png",
            title=f"MNIST λ={lam:g}: nonlinearity composition",
        )

        widths = {
            site: int(act.mode.shape[0])
            for site, act in zip(model.site_names, model.activations, strict=True)
        }
        selection = [s for s in stats if s.entropy <= LOW_ENTROPY_NATS]
        masked = copy.deepcopy(model)
        apply_modes(
            masked, assign_modes(selection, tie_break="identity", widths=widths)
        )
        folded = trim_folded(fold_mlp(masked))
        fstats = folded_stats(folded)
        print(f"mnist λ={lam:g}: surviving ReLUs {fstats.nonlinear_widths}")

        if fstats.nonlinear_widths[0] > 0:
            plot_input_filters(
                folded.stages[0].weight,
                (28, 28),
                OUT_DIR / f"mnist_surviving_filters_lam{lam:g}.png",
                title=(
                    f"MNIST λ={lam:g}: input filters of the "
                    f"{fstats.nonlinear_widths[0]} surviving layer-1 ReLUs"
                ),
            )
        if sum(fstats.nonlinear_widths) == 0:
            # Fully folded: the whole network is one affine map over the input.
            plot_input_filters(
                folded.head.weight,
                (28, 28),
                OUT_DIR / f"mnist_affine_templates_lam{lam:g}.png",
                title=f"MNIST λ={lam:g}: the folded network IS this affine map "
                "(per-class templates)",
            )
        elif sum(fstats.nonlinear_widths) <= 64:
            # The intensional program view: the folded network as a circuit DAG.
            plot_folded_dag(
                folded,
                OUT_DIR / f"mnist_dag_lam{lam:g}.png",
                image_shape=(28, 28),
                title=(
                    f"MNIST λ={lam:g}: the folded network as a circuit "
                    f"({sum(fstats.nonlinear_widths)} surviving ReLUs)"
                ),
            )

        # Class-indexed firing: which classes does each unit fire for?
        per_class = class_conditional_q(model, val_loader, base.train.device, 10)
        plot_class_q_matrix(
            per_class["relu0"],
            OUT_DIR / f"mnist_class_q_relu0_lam{lam:g}.png",
            title=f"MNIST λ={lam:g}, relu0: per-class firing rate (all units)",
        )
        survivors = (
            (masked.activations[0].mode == int(ActivationMode.RELU)).nonzero().flatten()
        )
        if 0 < survivors.numel() < 64:
            plot_class_q_matrix(
                per_class["relu0"][:, survivors],
                OUT_DIR / f"mnist_class_q_survivors_lam{lam:g}.png",
                title=(
                    f"MNIST λ={lam:g}: per-class firing of the "
                    f"{survivors.numel()} surviving relu0 units"
                ),
            )


def cnn_gallery() -> None:
    payload = json.loads(CNN_RESULTS.read_text())
    for lam in CNN_LAMBDAS:
        run = next(r for r in payload["runs"] if r["lam"] == lam)
        offset = 0
        for site, channels, side in CNN_SITES:
            n = channels * side * side
            q = torch.tensor(run["q"][offset : offset + n])
            offset += n
            plot_spatial_q_map(
                q,
                side,
                OUT_DIR / f"cnn_qmap_{site}_lam{lam:g}.png",
                max_channels=16,
                title=f"CIFAR-10 CNN λ={lam:g}, {site}: per-position hard q",
            )


def cnn_class_gallery() -> None:
    """Per-CLASS spatial q-maps for hand-picked conv0 channels at λ=10.

    Probes whether a "switching" channel (uniform q ≈ 0.5 in the aggregate
    map) actually switches *between classes* — i.e. is class-selective — or
    switches within every class.
    """
    base = config_from_json(Path("configs/cifar10-cnn.json"))
    _, val_loader, _ = make_dataloaders(base)
    model = build_model(base.model)
    load_checkpoint(_final_checkpoint(Path("runs") / f"{base.name}-10.0"), model)
    per_class = class_conditional_q(model, val_loader, base.train.device, 10)
    conv0 = per_class["conv0"]  # [10 classes, 32*32*32 positions]
    for ch in CNN_CLASS_CHANNELS:
        plot_spatial_q_map(
            conv0[:, ch * 1024 : (ch + 1) * 1024],
            side=32,
            path=OUT_DIR / f"cnn_class_qmap_conv0_ch{ch}_lam10.png",
            max_channels=10,
            title=f"CIFAR-10 CNN λ=10, conv0 ch{ch}: per-CLASS hard q",
            panel_labels=CIFAR_CLASSES,
        )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mnist_gallery()
    cnn_gallery()
    cnn_class_gallery()
    for path in sorted(OUT_DIR.iterdir()):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
