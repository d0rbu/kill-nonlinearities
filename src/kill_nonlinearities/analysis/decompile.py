"""Decompile a ReLUMLP over an input box into nested conditionals (phase 5).

Within a region where every earlier ReLU's sign is fixed, each unit's
pre-activation is an *affine function of the input*, so an undetermined ReLU
becomes a branch on an **input-space hyperplane** and the recursion yields a
binary decision tree whose leaves are affine maps — the network's exact
piecewise-linear semantics restricted to the box, rendered as a program:

    if w·x + b > 0:        # relu0[3] fires
        ...
    else:
        logits = A x + c

Two sign oracles build the same tree type:

- **Box-driven** (``decompile_mlp``): per region, each unit's sign is decided
  by (1) a closed-form range of its affine map over the box (free), then (2)
  an exact LP over the region polytope (box ∩ accumulated half-spaces, HiGHS),
  and only then (3) a branch. Leaves are exact on their *entire* region.
- **Data-driven** (``decompile_mlp_data``): the sign over a region is read off
  the region's actual samples, so the tree enumerates only the linear regions
  the data occupies. Leaves are exact for every building sample; held-out
  agreement is an empirical question.

``max_leaves`` bounds the enumeration; exhausted regions become ``Truncated``
nodes (reported, never silently dropped). Assigned ``ZERO``/``IDENTITY`` modes
are honored as exact affine behavior.

Everything is float64. LP-decided signs are widened by the same solver-slack
rule as ``analysis.ranges``, so leaf affine maps are exact and branch
predicates are sound up to HiGHS tolerances (property-tested by evaluating
trees against the network).
"""

from dataclasses import dataclass

import numpy as np
import torch
from scipy.optimize import linprog
from torch import Tensor

from kill_nonlinearities.models.activations import ActivationMode
from kill_nonlinearities.models.mlp import ReLUMLP

_LP_SLACK = 1e-7

__all__ = [
    "Branch",
    "Leaf",
    "TreeStats",
    "Truncated",
    "decompile_mlp",
    "decompile_mlp_data",
    "evaluate_tree",
    "render_tree",
    "route_leaves",
    "tree_stats",
]


@dataclass(frozen=True)
class Leaf:
    """An input region on which the network IS this affine map."""

    weight: Tensor  # [out, d]
    bias: Tensor  # [out]


@dataclass(frozen=True)
class Truncated:
    """A region left unresolved because the leaf budget ran out."""


@dataclass(frozen=True)
class Branch:
    """A sign test on one unit's (input-affine) pre-activation."""

    site: str
    index: int
    weight: Tensor  # [d]: test weight·x + bias > 0
    bias: float
    low: "Node"  # the unit is dead here (z ≤ 0)
    high: "Node"  # the unit is the identity here (z > 0)


Node = Branch | Leaf | Truncated


@dataclass(frozen=True)
class TreeStats:
    n_branches: int
    n_leaves: int
    n_truncated: int
    depth: int


class _Budget:
    def __init__(self, max_leaves: int) -> None:
        if max_leaves < 1:
            raise ValueError(f"max_leaves must be >= 1, got {max_leaves}")
        self.remaining = max_leaves

    def take_leaf(self) -> bool:
        if self.remaining < 1:
            return False
        self.remaining -= 1
        return True


def decompile_mlp(
    model: ReLUMLP, lower: Tensor, upper: Tensor, max_leaves: int = 256
) -> Node:
    """Exact piecewise-affine decompilation of ``model`` over the box."""
    lo = lower.detach().flatten().double()
    hi = upper.detach().flatten().double()
    if lo.shape != hi.shape:
        raise ValueError(f"box shapes differ: {lo.shape} vs {hi.shape}")
    if bool((lo > hi).any()):
        raise ValueError("box has lower > upper coordinates")

    layers = [
        (
            site,
            linear.weight.detach().double(),
            linear.bias.detach().double(),
            act.mode.clone(),
        )
        for site, linear, act in zip(
            model.site_names, model.linears, model.activations, strict=True
        )
    ]
    head_w = model.head.weight.detach().double()
    head_b = model.head.bias.detach().double()
    budget = _Budget(max_leaves)
    d = int(lo.shape[0])
    identity = torch.eye(d, dtype=torch.float64)
    zeros = torch.zeros(d, dtype=torch.float64)
    return _walk(
        layers,
        (head_w, head_b),
        layer_idx=0,
        start_index=0,
        act_m=identity,
        act_c=zeros,
        pre_m=None,
        pre_c=None,
        half_g=[],
        half_h=[],
        box=(lo.numpy(), hi.numpy()),
        budget=budget,
    )


def _walk(
    layers: list[tuple[str, Tensor, Tensor, Tensor]],
    head: tuple[Tensor, Tensor],
    layer_idx: int,
    start_index: int,
    act_m: Tensor,
    act_c: Tensor,
    pre_m: Tensor | None,
    pre_c: Tensor | None,
    half_g: list[np.ndarray],
    half_h: list[float],
    box: tuple[np.ndarray, np.ndarray],
    budget: _Budget,
) -> Node:
    """Resume the layer walk at (layer_idx, start_index) within one region.

    ``act_m``/``act_c`` is the affine map of the PREVIOUS layer's activations
    (the raw input at layer 0); ``pre_m``/``pre_c`` is this layer's affine
    pre-activation map, already built when resuming mid-layer after a branch.
    Rows of ``act_m`` for already-processed units of the current layer are
    rewritten in place on a cloned tensor as their signs resolve.
    """
    while layer_idx < len(layers):
        site, weight, bias, mode = layers[layer_idx]
        if pre_m is None:
            pre_m = weight @ act_m
            pre_c = weight @ act_c + bias
            act_m = pre_m.clone()
            act_c = pre_c.clone()
        assert pre_c is not None
        for j in range(start_index, int(weight.shape[0])):
            mode_j = int(mode[j])
            if mode_j == int(ActivationMode.ZERO):
                act_m[j] = 0.0
                act_c[j] = 0.0
                continue
            if mode_j == int(ActivationMode.IDENTITY):
                continue  # act rows already equal the pre-activation map
            sign = _region_sign(pre_m[j], float(pre_c[j]), half_g, half_h, box)
            if sign < 0:
                act_m[j] = 0.0
                act_c[j] = 0.0
            elif sign == 0:
                if budget.remaining < 2:
                    return Truncated()
                g = pre_m[j].numpy().astype(np.float64)
                c0 = float(pre_c[j])
                low_m = act_m.clone()
                low_c = act_c.clone()
                low_m[j] = 0.0
                low_c[j] = 0.0
                low = _walk(
                    layers,
                    head,
                    layer_idx,
                    j + 1,
                    low_m,
                    low_c,
                    pre_m,
                    pre_c,
                    [*half_g, g],
                    [*half_h, -c0],
                    box,
                    budget,
                )
                high = _walk(
                    layers,
                    head,
                    layer_idx,
                    j + 1,
                    act_m.clone(),
                    act_c.clone(),
                    pre_m,
                    pre_c,
                    [*half_g, -g],
                    [*half_h, c0],
                    box,
                    budget,
                )
                return Branch(
                    site=site,
                    index=j,
                    weight=pre_m[j].clone(),
                    bias=c0,
                    low=low,
                    high=high,
                )
        layer_idx += 1
        start_index = 0
        pre_m = None
        pre_c = None
    if not budget.take_leaf():
        return Truncated()
    head_w, head_b = head
    return Leaf(weight=head_w @ act_m, bias=head_w @ act_c + head_b)


def _region_sign(
    weight: Tensor,
    bias: float,
    half_g: list[np.ndarray],
    half_h: list[float],
    box: tuple[np.ndarray, np.ndarray],
) -> int:
    """-1 if z ≤ 0 on the whole region, +1 if z > 0, 0 if undetermined.

    Closed-form box range first (free; the region is inside the box); the LP
    over the full region polytope only runs when the box range straddles zero.
    """
    w = weight.numpy()
    lo, hi = box
    w_pos = np.clip(w, 0.0, None)
    w_neg = np.clip(w, None, 0.0)
    box_lo = float(w_pos @ lo + w_neg @ hi) + bias
    box_hi = float(w_pos @ hi + w_neg @ lo) + bias
    if box_lo > 0.0:
        return 1
    if box_hi <= 0.0:
        return -1
    if not half_g:
        return 0
    a_ub = np.vstack(half_g)
    b_ub = np.asarray(half_h)
    bounds = list(zip(lo, hi, strict=True))
    low = _solve(w, a_ub, b_ub, bounds) + bias
    if low - _LP_SLACK * (1.0 + abs(low)) > 0.0:
        return 1
    high = -_solve(-w, a_ub, b_ub, bounds) + bias
    if high + _LP_SLACK * (1.0 + abs(high)) <= 0.0:
        return -1
    return 0


def _solve(
    cost: np.ndarray,
    a_ub: np.ndarray,
    b_ub: np.ndarray,
    bounds: list[tuple[float, float]],
) -> float:
    result = linprog(cost, A_ub=a_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if result.status == 2:
        # Infeasible region (an empty intersection of earlier branches): no
        # input reaches it, so either sign claim is vacuously sound.
        return float("inf") if cost.any() else 0.0
    if result.status != 0:
        raise RuntimeError(f"region LP failed with status {result.status}")
    return float(result.fun)


def decompile_mlp_data(model: ReLUMLP, x: Tensor, max_leaves: int = 4096) -> Node:
    """Data-driven decompilation: branch only where the DATA flips a unit.

    Builds the tree from the sample set ``x`` instead of an input box: within
    a region (the samples selected by the branch path), a unit whose
    pre-activation keeps one sign across the region's samples is folded as
    stable, and the recursion branches only on units the data actually flips
    there — so the leaves are exactly the linear regions the data occupies
    (up to units that never needed a branch). Leaf affine maps are **exact
    for every building sample** routed to them (tested). A *fresh* input that
    follows the same branch path gets the leaf's affine map, which is exact
    iff its non-branched units share the region's signs — the agreement rate
    of that assumption on held-out data is an experiment, not a guarantee.
    ``ZERO``/``IDENTITY`` modes are honored exactly as in ``decompile_mlp``.
    """
    xs = x.detach().flatten(1).double()
    if xs.shape[0] == 0:
        raise ValueError("decompile_mlp_data requires at least one sample")
    layers = [
        (
            site,
            linear.weight.detach().double(),
            linear.bias.detach().double(),
            act.mode.clone(),
        )
        for site, linear, act in zip(
            model.site_names, model.linears, model.activations, strict=True
        )
    ]
    head = (model.head.weight.detach().double(), model.head.bias.detach().double())
    d = int(xs.shape[1])
    return _walk_data(
        layers,
        head,
        layer_idx=0,
        start_index=0,
        act_m=torch.eye(d, dtype=torch.float64),
        act_c=torch.zeros(d, dtype=torch.float64),
        pre_m=None,
        pre_c=None,
        xs=xs,
        z_layer=None,
        budget=_Budget(max_leaves),
    )


def _walk_data(
    layers: list[tuple[str, Tensor, Tensor, Tensor]],
    head: tuple[Tensor, Tensor],
    layer_idx: int,
    start_index: int,
    act_m: Tensor,
    act_c: Tensor,
    pre_m: Tensor | None,
    pre_c: Tensor | None,
    xs: Tensor,
    z_layer: Tensor | None,
    budget: _Budget,
) -> Node:
    """The resume-style walk of ``_walk``, with the data as the sign oracle.

    Same affine bookkeeping; a unit's sign over the region is read off the
    region's samples instead of box ranges/LPs, and branching splits the
    sample set (both sides are non-empty by construction). ``z_layer`` caches
    the region samples' pre-activations for the whole current layer — one GEMM
    per (region, layer) instead of a dot product per unit, which is what makes
    dataset-scale builds tractable; children inherit row slices of it.
    """
    while layer_idx < len(layers):
        site, weight, bias, mode = layers[layer_idx]
        if pre_m is None:
            pre_m = weight @ act_m
            pre_c = weight @ act_c + bias
            act_m = pre_m.clone()
            act_c = pre_c.clone()
        assert pre_c is not None
        if z_layer is None:
            z_layer = xs @ pre_m.T + pre_c
        for j in range(start_index, int(weight.shape[0])):
            mode_j = int(mode[j])
            if mode_j == int(ActivationMode.ZERO):
                act_m[j] = 0.0
                act_c[j] = 0.0
                continue
            if mode_j == int(ActivationMode.IDENTITY):
                continue
            fired = z_layer[:, j] > 0
            if bool(fired.all()):
                continue  # identity on this region's data
            if not bool(fired.any()):
                act_m[j] = 0.0
                act_c[j] = 0.0
                continue
            if budget.remaining < 2:
                return Truncated()
            low_m = act_m.clone()
            low_c = act_c.clone()
            low_m[j] = 0.0
            low_c[j] = 0.0
            low = _walk_data(
                layers,
                head,
                layer_idx,
                j + 1,
                low_m,
                low_c,
                pre_m,
                pre_c,
                xs[~fired],
                z_layer[~fired],
                budget,
            )
            high = _walk_data(
                layers,
                head,
                layer_idx,
                j + 1,
                act_m.clone(),
                act_c.clone(),
                pre_m,
                pre_c,
                xs[fired],
                z_layer[fired],
                budget,
            )
            return Branch(
                site=site,
                index=j,
                weight=pre_m[j].clone(),
                bias=float(pre_c[j]),
                low=low,
                high=high,
            )
        layer_idx += 1
        start_index = 0
        pre_m = None
        pre_c = None
        z_layer = None  # next layer's pre-acts depend on the resolved rows
    if not budget.take_leaf():
        return Truncated()
    head_w, head_b = head
    return Leaf(weight=head_w @ act_m, bias=head_w @ act_c + head_b)


def route_leaves(node: Node, x: Tensor) -> list[Node]:
    """The terminal node (``Leaf``/``Truncated``) each sample routes to.

    Consistent with ``evaluate_tree``'s branching (``z > 0`` goes high); unlike
    ``evaluate_tree`` it does not raise on ``Truncated`` — callers can mask.
    """
    xs = x.detach().flatten(1).double()
    out: list[Node] = [node] * int(xs.shape[0])

    def _route(current: Node, idx: Tensor) -> None:
        if isinstance(current, Branch):
            z = xs[idx] @ current.weight + current.bias
            high = z > 0
            if bool((~high).any()):
                _route(current.low, idx[~high])
            if bool(high.any()):
                _route(current.high, idx[high])
            return
        for i in idx.tolist():
            out[i] = current

    _route(node, torch.arange(int(xs.shape[0])))
    return out


def evaluate_tree(node: Node, x: Tensor) -> Tensor:
    """Evaluate the decompiled program on a batch (float64); exact by design."""
    x = x.flatten(1).double()
    if isinstance(node, Truncated):
        raise ValueError("input reached a truncated region")
    if isinstance(node, Leaf):
        return x @ node.weight.T + node.bias
    z = x @ node.weight + node.bias
    high_mask = z > 0
    out: Tensor | None = None
    for mask, child in ((~high_mask, node.low), (high_mask, node.high)):
        if not bool(mask.any()):
            continue
        result = evaluate_tree(child, x[mask])
        if out is None:
            out = torch.empty((x.shape[0], result.shape[1]), dtype=result.dtype)
        out[mask] = result
    assert out is not None  # x is non-empty: one side matched
    return out


def tree_stats(node: Node) -> TreeStats:
    if isinstance(node, Leaf):
        return TreeStats(n_branches=0, n_leaves=1, n_truncated=0, depth=0)
    if isinstance(node, Truncated):
        return TreeStats(n_branches=0, n_leaves=0, n_truncated=1, depth=0)
    low = tree_stats(node.low)
    high = tree_stats(node.high)
    return TreeStats(
        n_branches=low.n_branches + high.n_branches + 1,
        n_leaves=low.n_leaves + high.n_leaves,
        n_truncated=low.n_truncated + high.n_truncated,
        depth=max(low.depth, high.depth) + 1,
    )


def render_tree(node: Node, max_terms: int = 4, _indent: int = 0) -> str:
    """Printable pseudo-code: hyperplane tests with the largest-|coef| terms."""
    pad = "    " * _indent
    if isinstance(node, Truncated):
        return f"{pad}<truncated: leaf budget exhausted>"
    if isinstance(node, Leaf):
        out, d = node.weight.shape
        return f"{pad}return affine({out}x{d}) @ x + bias"
    terms = _hyperplane_terms(node.weight, node.bias, max_terms)
    return (
        f"{pad}if {terms} > 0:  # {node.site}[{node.index}] fires\n"
        f"{render_tree(node.high, max_terms, _indent + 1)}\n"
        f"{pad}else:\n"
        f"{render_tree(node.low, max_terms, _indent + 1)}"
    )


def _hyperplane_terms(weight: Tensor, bias: float, max_terms: int) -> str:
    order = torch.argsort(weight.abs(), descending=True)
    shown = [f"{float(weight[i]):+.3f}*x[{int(i)}]" for i in order[:max_terms].tolist()]
    if int(weight.shape[0]) > max_terms:
        shown.append("…")
    return " ".join(shown) + f" {bias:+.3f}"
