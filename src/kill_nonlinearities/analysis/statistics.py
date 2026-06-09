"""Hard-sign statistics over a model's pre-activations (spec §4.10).

``collect_pre_activations`` uses the exact ``model(x).pre_activations`` path that
surgery eval also uses, so analysis and surgery stay bit-exact (I2, [R7]).
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from kill_nonlinearities.models.mlp import ReLUMLP
from kill_nonlinearities.regularization.entropy import (
    hard_fraction_positive,
    sign_entropy,
)
from kill_nonlinearities.training.checkpoint import load_checkpoint


def collect_pre_activations(
    model: ReLUMLP, loader: DataLoader, device: str
) -> dict[str, Tensor]:
    """Concatenate per-site pre-activations over ``loader`` (spec §4.10).

    Runs ``model.eval()`` under ``torch.no_grad()`` and uses the exact
    ``ForwardOutput.pre_activations`` tensors (same forward path as surgery eval).
    """
    model.eval()
    chunks: dict[str, list[Tensor]] = {}
    site_names: tuple[str, ...] = ()
    with torch.no_grad():
        for x, _ in loader:
            out = model(x.to(device))
            site_names = out.site_names
            for site, z in zip(out.site_names, out.pre_activations, strict=True):
                chunks.setdefault(site, []).append(z)
    return {site: torch.cat(chunks[site], dim=0) for site in site_names}


@dataclass(frozen=True)
class NeuronStats:
    site: str
    index: int
    q: float
    entropy: float
    mean_pre: float


def neuron_stats(pre_by_site: dict[str, Tensor]) -> list[NeuronStats]:
    """Per-neuron hard fraction-positive, sign-entropy, and mean pre-activation."""
    stats: list[NeuronStats] = []
    for site, pre in pre_by_site.items():
        q = hard_fraction_positive(pre)
        entropy = sign_entropy(q)
        mean_pre = pre.mean(dim=0)
        for index in range(pre.shape[1]):
            stats.append(
                NeuronStats(
                    site=site,
                    index=index,
                    q=float(q[index]),
                    entropy=float(entropy[index]),
                    mean_pre=float(mean_pre[index]),
                )
            )
    return stats


@dataclass(frozen=True)
class FrameStats:
    step: int
    probe_activations: dict[tuple[str, int], Tensor]
    q_by_site: dict[str, Tensor]


def collect_history(
    checkpoint_paths: Sequence[Path],
    model_factory: Callable[[], ReLUMLP],
    probe_batch: Tensor,
    val_loader: DataLoader,
    probe_neurons: Sequence[tuple[str, int]],
    device: str,
) -> list[FrameStats]:
    """One ``FrameStats`` per checkpoint (spec §4.10, [R11]).

    For each checkpoint: rebuild the model, strict-load the state, then capture
    the probe neurons' pre-activations on the fixed ``probe_batch`` and the hard
    ``q_i`` over ``val_loader``.
    """
    frames: list[FrameStats] = []
    for path in checkpoint_paths:
        # Own the device move so the rebuilt model matches the device tensors
        # (``probe_batch.to(device)`` / ``collect_pre_activations(..., device)``)
        # regardless of what ``model_factory`` returns (capstone fix).
        model = model_factory().to(device)
        step = load_checkpoint(path, model)
        model.eval()
        with torch.no_grad():
            out = model(probe_batch.to(device))
        site_to_pos = {site: i for i, site in enumerate(out.site_names)}
        probe_activations: dict[tuple[str, int], Tensor] = {}
        for site, index in probe_neurons:
            z = out.pre_activations[site_to_pos[site]]
            probe_activations[site, index] = z[:, index]
        pre_by_site = collect_pre_activations(model, val_loader, device)
        q_by_site = {
            site: hard_fraction_positive(pre) for site, pre in pre_by_site.items()
        }
        frames.append(
            FrameStats(
                step=step,
                probe_activations=probe_activations,
                q_by_site=q_by_site,
            )
        )
    return frames
