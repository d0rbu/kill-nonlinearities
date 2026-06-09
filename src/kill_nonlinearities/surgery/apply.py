"""Masked-activation surgery: apply modes, evaluate, and run the k-sweep (§4.11)."""

import copy
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from kill_nonlinearities.analysis.selection import (
    assign_modes,
    select_topk,
)
from kill_nonlinearities.analysis.statistics import NeuronStats
from kill_nonlinearities.models.mlp import ReLUMLP


def apply_modes(model: ReLUMLP, modes_by_site: dict[str, Tensor]) -> None:
    """Set each ``SelectiveReLU``'s mode buffer from a full ``[N]`` tensor (§4.11)."""
    site_to_act = dict(zip(model.site_names, model.activations, strict=True))
    for site, modes in modes_by_site.items():
        site_to_act[site].set_modes(modes)


def evaluate_accuracy(model: nn.Module, loader: DataLoader, device: str) -> float:
    """Top-1 accuracy over ``loader`` (spec §4.11)."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device)).logits
            preds = logits.argmax(dim=1)
            correct += int((preds == y.to(device)).sum())
            total += int(y.shape[0])
    return correct / total if total > 0 else 0.0


@dataclass(frozen=True)
class KPoint:
    k: int
    val_acc: float
    test_acc: float


def k_sweep(
    model: ReLUMLP,
    ranked: list[NeuronStats],
    k_grid: list[int],
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: str,
) -> list[KPoint]:
    """Accuracy-vs-k on val and test (spec §4.11, [R5][R21]).

    Works on a single ``copy.deepcopy(model)``; the canonical ``model`` is never
    mutated. ``assign_modes`` receives the true per-site widths so unselected
    neurons stay ``RELU``.
    """
    work = copy.deepcopy(model)
    widths = {
        site: int(act.mode.shape[0])
        for site, act in zip(model.site_names, model.activations, strict=True)
    }
    points: list[KPoint] = []
    for k in k_grid:
        modes = assign_modes(
            select_topk(ranked, k), tie_break="identity", widths=widths
        )
        apply_modes(work, modes)
        points.append(
            KPoint(
                k=k,
                val_acc=evaluate_accuracy(work, val_loader, device),
                test_acc=evaluate_accuracy(work, test_loader, device),
            )
        )
    return points
