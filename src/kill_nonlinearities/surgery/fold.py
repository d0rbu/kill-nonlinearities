"""Structural surgery: fold a mode-assigned ReLUMLP into a smaller network (phase 2).

``fold_mlp`` converts a ``ReLUMLP`` whose ``SelectiveReLU`` modes were assigned
by analysis into a ``FoldedMLP`` computing the same function with the masked
nonlinearities removed from the architecture itself: ``ZERO`` neurons are
dropped (their rows and consumer columns deleted), ``IDENTITY`` neurons are
pre-multiplied into the next layer's weights as an affine bypass, and only
``RELU`` neurons survive as actual nonlinearities. The fold is exact in real
arithmetic; float32 evaluation agrees with the masked model up to reduction
reassociation (ULP-level — covered by float64 tests).

The folded forward keeps an *augmented* feature vector: at stage ``l``,
``u_{l+1} = [relu(W_l u_l + b_l), u_l[carry_l]]``. Identity neurons' affine
maps are composed into consumer weights *at fold time*, so a fully-``IDENTITY``
network collapses into a single affine map, and a fully-``ZERO`` layer leaves
only the bias path. ``trim_folded`` then removes every coordinate that no
consumer reads (zero columns, propagated backward through the carries), which
(a) reduces the carry to nothing when there are no ``IDENTITY`` neurons —
pure width pruning yields a plain smaller MLP — and (b) drops ``RELU`` units
whose outputs are never read downstream.
"""

from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor, nn

from kill_nonlinearities.models.activations import ActivationMode
from kill_nonlinearities.models.mlp import ReLUMLP


class FoldedMLP(nn.Module):
    """Augmented-state affine/ReLU chain produced by ``fold_mlp``.

    Stage ``l`` holds a ``Linear(d_l, r_l)`` (its ReLU part) and an int64
    ``carry_l`` index buffer selecting which coordinates of ``u_l`` are carried
    forward: ``u_{l+1} = [relu(stage_l(u_l)), u_l[carry_l]]``. The ``head`` is
    the final affine over ``u_L``. Logits are returned directly (this is an
    inference artifact: no pre-activation sites remain to emit).
    """

    def __init__(
        self,
        stages: list[nn.Linear],
        carries: list[Tensor],
        head: nn.Linear,
    ) -> None:
        super().__init__()
        if len(stages) != len(carries):
            raise ValueError(
                f"{len(stages)} stages but {len(carries)} carry index tensors"
            )
        self._stages = nn.ModuleList(stages)
        self.head = head
        for i, carry in enumerate(carries):
            if carry.dtype != torch.int64:
                raise ValueError(f"carry {i} must be int64, got {carry.dtype}")
            self.register_buffer(f"_carry_{i}", carry)

    @property
    def stages(self) -> list[nn.Linear]:
        return cast(list[nn.Linear], list(self._stages))

    @property
    def carries(self) -> list[Tensor]:
        return [getattr(self, f"_carry_{i}") for i in range(len(self._stages))]

    def forward(self, x: Tensor) -> Tensor:
        u = x.flatten(1)
        for stage, carry in zip(self._stages, self.carries, strict=True):
            u = torch.cat([torch.relu(stage(u)), u[:, carry]], dim=1)
        return self.head(u)


@dataclass(frozen=True)
class FoldStats:
    """Structural summary of a fold (per-stage ReLU widths + parameter counts)."""

    nonlinear_widths: tuple[int, ...]
    carry_widths: tuple[int, ...]
    params: int


def folded_stats(folded: FoldedMLP) -> FoldStats:
    """ReLU width / carry width per stage and the total parameter count."""
    return FoldStats(
        nonlinear_widths=tuple(s.out_features for s in folded.stages),
        carry_widths=tuple(int(c.shape[0]) for c in folded.carries),
        params=sum(p.numel() for p in folded.parameters()),
    )


def _as_linear(weight: Tensor, bias: Tensor) -> nn.Linear:
    # Preserve the source dtype: folding a float64 model must compose its
    # weight products in float64 (exactness tests rely on this).
    linear = nn.Linear(weight.shape[1], weight.shape[0], dtype=weight.dtype)
    with torch.no_grad():
        linear.weight.copy_(weight)
        linear.bias.copy_(bias)
    return linear


def fold_mlp(model: ReLUMLP) -> FoldedMLP:
    """Fold ``model``'s assigned modes into the architecture (spec in module doc).

    Walks the hidden layers keeping an *effective* affine map of the current
    layer over the augmented input ``u_l``: ``RELU`` rows become the stage's
    ReLU part, ``IDENTITY`` rows are composed into the next layer's effective
    weights (landing on the carried-``u_l`` block), and ``ZERO`` rows vanish.
    The untouched ``model`` is never mutated.
    """
    relu_code = int(ActivationMode.RELU)
    id_code = int(ActivationMode.IDENTITY)

    linears = model.linears
    nexts = [*linears[1:], model.head]

    stages: list[nn.Linear] = []
    carries: list[Tensor] = []
    with torch.no_grad():
        eff_weight = linears[0].weight.detach().clone()
        eff_bias = linears[0].bias.detach().clone()
        for act, nxt in zip(model.activations, nexts, strict=True):
            keep = act.mode == relu_code
            ident = act.mode == id_code
            d = eff_weight.shape[1]

            stages.append(_as_linear(eff_weight[keep], eff_bias[keep]))
            carries.append(torch.arange(d, dtype=torch.int64))

            # The next layer's affine over u_{l+1} = [relu part, u_l]:
            # RELU columns pass through; IDENTITY columns compose with this
            # layer's effective affine; ZERO columns are dropped.
            v_weight = nxt.weight.detach()
            eff_weight = torch.cat(
                [v_weight[:, keep], v_weight[:, ident] @ eff_weight[ident]], dim=1
            )
            eff_bias = nxt.bias.detach() + v_weight[:, ident] @ eff_bias[ident]

    return FoldedMLP(stages, carries, _as_linear(eff_weight, eff_bias))


def trim_folded(folded: FoldedMLP) -> FoldedMLP:
    """Drop every coordinate no consumer reads (backward zero-column pass).

    A coordinate of ``u_{l+1}`` is *needed* if any downstream weight column
    that reads it is nonzero. Backward from the head: unneeded ReLU units are
    deleted, carries shrink to the coordinates actually read later, and the
    surviving weights' columns are compacted to match. The input ``u_0 = x``
    is never trimmed (the folded model keeps the original input signature).
    Function-preserving: dropped coordinates only ever met zero weights.
    """
    stages = folded.stages
    carries = folded.carries
    n = len(stages)

    with torch.no_grad():
        # Backward pass: needed[l] is a bool mask over u_l's coordinates.
        needed: list[Tensor] = [torch.empty(0)] * (n + 1)
        needed[n] = folded.head.weight.abs().sum(dim=0) > 0
        for i in range(n - 1, -1, -1):
            r = stages[i].out_features
            h_keep = needed[i + 1][:r]
            carry_keep = needed[i + 1][r:]
            d = stages[i].in_features
            used = torch.zeros(d, dtype=torch.bool)
            if bool(h_keep.any()):
                used |= stages[i].weight[h_keep].abs().sum(dim=0) > 0
            used[carries[i][carry_keep]] = True
            needed[i] = used
        needed[0] = torch.ones(stages[0].in_features, dtype=torch.bool)

        # Forward rebuild: compact each u_l by needed[l]; remap carry indices.
        new_stages: list[nn.Linear] = []
        new_carries: list[Tensor] = []
        for i in range(n):
            r = stages[i].out_features
            h_keep = needed[i + 1][:r]
            carry_keep = needed[i + 1][r:]
            new_stages.append(
                _as_linear(
                    stages[i].weight[h_keep][:, needed[i]],
                    stages[i].bias[h_keep],
                )
            )
            # Positions of u_l coordinates inside the compacted u_l.
            position = torch.cumsum(needed[i].to(torch.int64), dim=0) - 1
            new_carries.append(position[carries[i][carry_keep]])

        new_head = _as_linear(
            folded.head.weight[:, needed[n]], folded.head.bias.detach().clone()
        )
    return FoldedMLP(new_stages, new_carries, new_head)
