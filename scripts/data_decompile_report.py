"""Phase-5 follow-up: DATA-driven decompilation + a linear-region census.

Two experiments answering "build the trees from the data, and how far can we
take this?":

1. **MNIST data-driven trees** (λ ∈ {0, 1, 10}): build the tree from a fixed
   subset of training samples (`decompile_mlp_data` — branch only where the
   data flips a unit), then measure (a) exactness on the building samples
   (must be ~1e-9), and (b) **held-out generalization** on the test set:
   argmax agreement with the network, and the fraction of test samples whose
   leaf affine map is NOT the network's local linearization for them
   ("novel-pattern rate" — they flip some unit the building data never
   branched on).
2. **Linear-region census** (MNIST MLP λ ∈ {0, 1, 10}; CIFAR-10 CNN
   λ ∈ {0, 10}): count the DISTINCT activation sign patterns the training
   split actually occupies — exactly the number of leaves a complete
   data-driven tree would need, i.e. "how big is the network's program *on
   its data*".

Writes ``docs/research/assets/data_decompile_results.json`` and
``data_decompile.png``.

Usage:
    uv run --no-sync python scripts/data_decompile_report.py
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from kill_nonlinearities.analysis.decompile import (
    Leaf,
    decompile_mlp_data,
    evaluate_tree,
    route_leaves,
    tree_stats,
)
from kill_nonlinearities.data.datasets import make_dataloaders
from kill_nonlinearities.experiments.run import config_from_json
from kill_nonlinearities.models import PreActModel, build_model
from kill_nonlinearities.training.checkpoint import load_checkpoint

ASSET_DIR = Path("docs/research/assets")
MNIST_LAMBDAS = (0.0, 1.0, 10.0)
CNN_LAMBDAS = (0.0, 10.0)
BUILD_SAMPLES = 8192
MAX_LEAVES = 4096


def _final_checkpoint(run_dir: Path) -> Path:
    steps = {
        int(m.group(1)): p
        for p in run_dir.glob("step_*.pt")
        if (m := re.fullmatch(r"step_(\d+)\.pt", p.name))
    }
    return steps[max(steps)]


def _collect(loader) -> tuple[torch.Tensor, torch.Tensor]:
    xs, ys = [], []
    for x, y in loader:
        xs.append(x.flatten(1))
        ys.append(y)
    return torch.cat(xs), torch.cat(ys)


def data_trees(config_path: str, lambdas: tuple[float, ...]) -> list[dict]:
    base = config_from_json(Path(config_path))
    train_loader, _, test_loader = make_dataloaders(base)
    x_train, _ = _collect(train_loader)
    x_build = x_train[:BUILD_SAMPLES].double()
    x_test, y_test = _collect(test_loader)
    x_test = x_test.double()

    records: list[dict] = []
    for lam in lambdas:
        model = build_model(base.model)
        load_checkpoint(_final_checkpoint(Path("runs") / f"{base.name}-{lam}"), model)
        model = model.double()
        model.eval()

        t0 = time.perf_counter()
        tree = decompile_mlp_data(model, x_build, max_leaves=MAX_LEAVES)
        seconds = time.perf_counter() - t0
        stats = tree_stats(tree)

        with torch.no_grad():
            net_build = model(x_build).logits
            net_test = model(x_test).logits

        # (a) Exactness on the building samples (leaves are their regions).
        build_ok = torch.tensor(
            [isinstance(leaf, Leaf) for leaf in route_leaves(tree, x_build)]
        )
        build_diff = float(
            (evaluate_tree(tree, x_build[build_ok]) - net_build[build_ok]).abs().max()
        )

        # (b) Held-out generalization of the program.
        test_ok = torch.tensor(
            [isinstance(leaf, Leaf) for leaf in route_leaves(tree, x_test)]
        )
        tree_test = evaluate_tree(tree, x_test[test_ok])
        net_ok = net_test[test_ok]
        agreement = float(
            (tree_test.argmax(dim=1) == net_ok.argmax(dim=1)).double().mean()
        )
        novel = float(((tree_test - net_ok).abs().amax(dim=1) > 1e-6).double().mean())
        test_acc_tree = float(
            (tree_test.argmax(dim=1) == y_test[test_ok]).double().mean()
        )
        record = {
            "experiment": "data_tree",
            "dataset": base.name,
            "lam": lam,
            "build_samples": int(x_build.shape[0]),
            "branches": stats.n_branches,
            "leaves": stats.n_leaves,
            "truncated": stats.n_truncated,
            "depth": stats.depth,
            "build_routed": int(build_ok.sum()),
            "build_max_diff": build_diff,
            "test_routed": int(test_ok.sum()),
            "test_total": int(x_test.shape[0]),
            "test_agreement": agreement,
            "test_novel_pattern_rate": novel,
            "test_acc_tree": test_acc_tree,
            "seconds": seconds,
        }
        records.append(record)
        print(
            f"{base.name} data-tree λ={lam:<4g}: leaves={stats.n_leaves:>4} "
            f"branches={stats.n_branches:>4} truncated={stats.n_truncated:>3} "
            f"depth={stats.depth:>3} buildΔ={build_diff:.1e} "
            f"test agree={agreement:.4f} novel={novel:.3f} "
            f"acc={test_acc_tree:.4f} ({seconds:.1f}s)",
            flush=True,
        )
    return records


def _census(model: PreActModel, loader, device: str) -> int:
    """Distinct activation sign patterns over ``loader`` (hashed, exact)."""
    model.eval()
    seen: set[bytes] = set()
    with torch.no_grad():
        for x, _ in loader:
            out = model(x.to(device))
            fired = torch.cat([(z > 0) for z in out.pre_activations], dim=1).numpy()
            packed = np.packbits(fired, axis=1)
            for row in packed:
                seen.add(hashlib.blake2b(row.tobytes(), digest_size=16).digest())
    return len(seen)


def region_census() -> list[dict]:
    records: list[dict] = []
    targets = (
        [("configs/mnist.json", lam) for lam in MNIST_LAMBDAS]
        + [("configs/cifar10.json", lam) for lam in MNIST_LAMBDAS]
        + [("configs/cifar10-cnn.json", lam) for lam in CNN_LAMBDAS]
    )
    for config_path, lam in targets:
        base = config_from_json(Path(config_path))
        train_loader, _, _ = make_dataloaders(base)
        model = build_model(base.model)
        load_checkpoint(_final_checkpoint(Path("runs") / f"{base.name}-{lam}"), model)
        n_samples = len(train_loader.dataset)  # ty: ignore[invalid-argument-type]
        t0 = time.perf_counter()
        regions = _census(model, train_loader, base.train.device)
        seconds = time.perf_counter() - t0
        records.append(
            {
                "experiment": "census",
                "dataset": base.name,
                "lam": lam,
                "train_samples": int(n_samples),
                "distinct_regions": regions,
                "seconds": seconds,
            }
        )
        print(
            f"census {base.name:<12} λ={lam:<4g}: {regions} distinct regions "
            f"over {n_samples} training samples ({seconds:.0f}s)",
            flush=True,
        )
    return records


def _plot(records: list[dict], path: Path) -> None:
    census = [r for r in records if r["experiment"] == "census"]
    datasets = sorted({r["dataset"] for r in census})
    fig, axes = plt.subplots(
        1, len(datasets), figsize=(5.5 * len(datasets), 4.2), squeeze=False
    )
    for ax, dataset in zip(axes[0], datasets, strict=True):
        rows = sorted(
            (r for r in census if r["dataset"] == dataset), key=lambda r: r["lam"]
        )
        xs = range(len(rows))
        ax.bar(xs, [r["distinct_regions"] for r in rows], color="tab:blue")
        ax.axhline(
            rows[0]["train_samples"],
            color="grey",
            linestyle=":",
            label="training samples (upper bound)",
        )
        ax.set_xticks(list(xs), [f"λ={r['lam']:g}" for r in rows])
        ax.set_yscale("log")
        ax.set_ylabel("distinct linear regions occupied by the data")
        ax.set_title(dataset)
        ax.legend(fontsize=8)
    fig.tight_layout()
    try:
        fig.savefig(path, dpi=110, bbox_inches="tight")
    finally:
        plt.close(fig)
    print(f"wrote {path}")


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    records = (
        data_trees("configs/mnist.json", MNIST_LAMBDAS)
        + data_trees("configs/cifar10.json", MNIST_LAMBDAS)
        + region_census()
    )
    (ASSET_DIR / "data_decompile_results.json").write_text(
        json.dumps({"runs": records}) + "\n"
    )
    _plot(records, ASSET_DIR / "data_decompile.png")


if __name__ == "__main__":
    main()
