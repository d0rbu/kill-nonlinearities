"""Pre-activation range analysis over an input box (phase 4).

Given a per-coordinate input box ``[lower, upper]`` and a (mode-aware)
``ReLUMLP``, bound every hidden neuron's pre-activation, layer by layer:

- ``interval_ranges`` — interval bound propagation (IBP). The per-neuron LP
  over a box has the closed-form optimum ``W⁺·u + W⁻·l + b``, applied
  recursively through the (mode-aware) activation. Cheap; loosens with depth.
- ``lp_ranges`` — two linear programs per neuron (``scipy`` / HiGHS simplex)
  over the input box, keeping every previous layer as linear constraints:
  sign-stable ReLUs and ``IDENTITY`` units are exact affine equalities,
  ``ZERO`` units are pinned at 0, and unstable ReLUs use the standard triangle
  relaxation (``a ≥ 0``, ``a ≥ z``, ``a ≤ u(z-l)/(u-l)``). Sound, at least as
  tight as IBP, and exact for the first layer (property-tested).

A neuron whose certified range excludes zero is sign-consistent for **every**
input in the box — a strictly stronger statement than the empirical hard ``q``
(which only speaks about the sampled data). ``certified_modes`` turns ranges
into per-site ``ActivationMode`` tensors directly consumable by surgery.

All arithmetic is float64. LP bounds are widened by a solver-tolerance slack
(``1e-7 · (1 + |bound|)``) so the soundness claim survives HiGHS's default
termination tolerances.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from scipy.optimize import linprog
from torch import Tensor

from kill_nonlinearities.models.activations import ActivationMode
from kill_nonlinearities.models.mlp import ReLUMLP

_LP_SLACK = 1e-7

__all__ = [
    "LayerRanges",
    "certified_modes",
    "data_box",
    "interval_ranges",
    "lp_ranges",
]


@dataclass(frozen=True)
class LayerRanges:
    """Per-neuron pre-activation bounds at one site (float64 tensors)."""

    site: str
    pre_lower: Tensor
    pre_upper: Tensor


def data_box(loader: torch.utils.data.DataLoader) -> tuple[Tensor, Tensor]:
    """Per-coordinate min/max of the (flattened) inputs over ``loader``."""
    lower: Tensor | None = None
    upper: Tensor | None = None
    for x, _ in loader:
        flat = x.flatten(1).double()
        batch_lo = flat.min(dim=0).values
        batch_hi = flat.max(dim=0).values
        lower = batch_lo if lower is None else torch.minimum(lower, batch_lo)
        upper = batch_hi if upper is None else torch.maximum(upper, batch_hi)
    if lower is None or upper is None:
        raise ValueError("data_box requires a non-empty loader")
    return lower, upper


def _validate_box(lower: Tensor, upper: Tensor) -> tuple[Tensor, Tensor]:
    lo = lower.detach().flatten().double()
    hi = upper.detach().flatten().double()
    if lo.shape != hi.shape:
        raise ValueError(f"box shapes differ: {lo.shape} vs {hi.shape}")
    if bool((lo > hi).any()):
        raise ValueError("box has lower > upper coordinates")
    return lo, hi


def _activation_bounds(
    mode: Tensor, pre_lower: Tensor, pre_upper: Tensor
) -> tuple[Tensor, Tensor]:
    """Post-activation bounds per neuron, honoring the assigned mode."""
    relu_lo = pre_lower.clamp(min=0.0)
    relu_hi = pre_upper.clamp(min=0.0)
    zero = torch.zeros_like(pre_lower)
    is_zero = mode == int(ActivationMode.ZERO)
    is_ident = mode == int(ActivationMode.IDENTITY)
    lo = torch.where(is_zero, zero, torch.where(is_ident, pre_lower, relu_lo))
    hi = torch.where(is_zero, zero, torch.where(is_ident, pre_upper, relu_hi))
    return lo, hi


def interval_ranges(model: ReLUMLP, lower: Tensor, upper: Tensor) -> list[LayerRanges]:
    """IBP pre-activation bounds for every hidden site (mode-aware)."""
    lo, hi = _validate_box(lower, upper)
    out: list[LayerRanges] = []
    with torch.no_grad():
        for site, linear, act in zip(
            model.site_names, model.linears, model.activations, strict=True
        ):
            weight = linear.weight.double()
            bias = linear.bias.double()
            w_pos = weight.clamp(min=0.0)
            w_neg = weight.clamp(max=0.0)
            pre_lo = w_pos @ lo + w_neg @ hi + bias
            pre_hi = w_pos @ hi + w_neg @ lo + bias
            out.append(LayerRanges(site=site, pre_lower=pre_lo, pre_upper=pre_hi))
            lo, hi = _activation_bounds(act.mode, pre_lo, pre_hi)
    return out


def lp_ranges(model: ReLUMLP, lower: Tensor, upper: Tensor) -> list[LayerRanges]:
    """LP pre-activation bounds for every hidden site (mode-aware, sound).

    Variables are the input coordinates plus one post-activation variable per
    previous neuron; constraints encode each previous layer exactly where
    possible (stable / ``IDENTITY`` / ``ZERO``) and via the triangle relaxation
    where the sign is uncertain. Two HiGHS solves per neuron.
    """
    lo, hi = _validate_box(lower, upper)
    var_lo: list[float] = lo.tolist()
    var_hi: list[float] = hi.tolist()
    eq_rows: list[np.ndarray] = []
    eq_vals: list[float] = []
    ub_rows: list[np.ndarray] = []
    ub_vals: list[float] = []
    prev_offset = 0
    prev_width = int(lo.shape[0])

    out: list[LayerRanges] = []
    with torch.no_grad():
        for site, linear, act in zip(
            model.site_names, model.linears, model.activations, strict=True
        ):
            weight = linear.weight.double().numpy()
            bias = linear.bias.double().numpy()
            n_neurons, n_vars = weight.shape[0], len(var_lo)
            a_ub = np.vstack(ub_rows) if ub_rows else None
            b_ub = np.asarray(ub_vals) if ub_rows else None
            a_eq = np.vstack(eq_rows) if eq_rows else None
            b_eq = np.asarray(eq_vals) if eq_rows else None
            bounds = list(zip(var_lo, var_hi, strict=True))

            pre_lo = np.empty(n_neurons)
            pre_hi = np.empty(n_neurons)
            for j in range(n_neurons):
                cost = np.zeros(n_vars)
                cost[prev_offset : prev_offset + prev_width] = weight[j]
                low = _solve(cost, a_ub, b_ub, a_eq, b_eq, bounds, site, j)
                high = -_solve(-cost, a_ub, b_ub, a_eq, b_eq, bounds, site, j)
                slack_lo = _LP_SLACK * (1.0 + abs(low + bias[j]))
                slack_hi = _LP_SLACK * (1.0 + abs(high + bias[j]))
                pre_lo[j] = low + bias[j] - slack_lo
                pre_hi[j] = high + bias[j] + slack_hi
            out.append(
                LayerRanges(
                    site=site,
                    pre_lower=torch.from_numpy(pre_lo),
                    pre_upper=torch.from_numpy(pre_hi),
                )
            )

            # Introduce this layer's post-activation variables + constraints.
            act_lo, act_hi = _activation_bounds(
                act.mode, torch.from_numpy(pre_lo), torch.from_numpy(pre_hi)
            )
            mode = act.mode
            a_offset = n_vars
            new_width = n_vars + n_neurons
            for j in range(n_neurons):
                var_lo.append(float(act_lo[j]))
                var_hi.append(float(act_hi[j]))
                mode_j = int(mode[j])
                affine = np.zeros(new_width)
                affine[prev_offset : prev_offset + prev_width] = weight[j]
                a_j = np.zeros(new_width)
                a_j[a_offset + j] = 1.0
                if mode_j == int(ActivationMode.ZERO):
                    continue  # variable bounds (0, 0) already pin it
                if mode_j == int(ActivationMode.IDENTITY) or pre_lo[j] >= 0.0:
                    eq_rows.append(a_j - affine)
                    eq_vals.append(float(bias[j]))
                elif pre_hi[j] <= 0.0:
                    continue  # stably negative ReLU: bounds are (0, 0)
                else:
                    # Triangle relaxation: a >= z and a <= slope * (z - lo).
                    ub_rows.append(affine - a_j)
                    ub_vals.append(float(-bias[j]))
                    slope = pre_hi[j] / (pre_hi[j] - pre_lo[j])
                    ub_rows.append(a_j - slope * affine)
                    ub_vals.append(float(slope * (bias[j] - pre_lo[j])))
            # Pad previously accumulated rows to the new variable width.
            eq_rows = [_pad(row, new_width) for row in eq_rows]
            ub_rows = [_pad(row, new_width) for row in ub_rows]
            prev_offset = a_offset
            prev_width = n_neurons
    return out


def _pad(row: np.ndarray, width: int) -> np.ndarray:
    if row.shape[0] == width:
        return row
    padded = np.zeros(width)
    padded[: row.shape[0]] = row
    return padded


def _solve(
    cost: np.ndarray,
    a_ub: np.ndarray | None,
    b_ub: np.ndarray | None,
    a_eq: np.ndarray | None,
    b_eq: np.ndarray | None,
    bounds: list[tuple[float, float]],
    site: str,
    index: int,
) -> float:
    result = linprog(
        cost,
        A_ub=a_ub,
        b_ub=b_ub,
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )
    if result.status != 0:
        raise RuntimeError(
            f"LP for {site}[{index}] failed with status {result.status}: "
            f"{result.message}"
        )
    return float(result.fun)


def certified_modes(ranges: Sequence[LayerRanges]) -> dict[str, Tensor]:
    """Per-site ``ActivationMode`` tensors from certified ranges.

    ``IDENTITY`` where the lower bound is strictly positive (the ReLU is
    provably the identity on the whole box), ``ZERO`` where the upper bound is
    ≤ 0 (provably dead), ``RELU`` otherwise. Directly consumable by
    ``surgery.apply.apply_modes`` / ``surgery.fold``.
    """
    modes: dict[str, Tensor] = {}
    for layer in ranges:
        mode = torch.full(
            (int(layer.pre_lower.shape[0]),),
            int(ActivationMode.RELU),
            dtype=torch.int64,
        )
        mode[layer.pre_upper <= 0.0] = int(ActivationMode.ZERO)
        mode[layer.pre_lower > 0.0] = int(ActivationMode.IDENTITY)
        modes[layer.site] = mode
    return modes
