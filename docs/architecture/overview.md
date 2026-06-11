# Architecture overview

[← Architecture](README.md) · [← Documentation hub](../README.md)

> **Status:** Phase 1a is **in progress**. The
> `regularization/`, `models/`, `training/`, `data/`, `analysis/`, `surgery/`, `viz/`, and
> `experiments/` packages exist. This document tracks the realized structure and the
> load-bearing design decisions.

## Guiding principles

1. **Small, single-purpose modules** with clear interfaces — each unit is understandable and
   testable on its own.
2. **Pure functions for the math.** The regularizer's pieces (soft sign, batch fraction,
   entropy, loss) are plain functions of tensors with no hidden state, so they are trivially
   unit- and property-tested. See [testing](../development/testing.md).
3. **Explicit over implicit** (see the key decision below).
4. **Analysis-first.** Build the measurement path before the surgery path; see the
   [roadmap](../research/README.md#roadmap).

## Key decision: models emit their own pre-activations (no hooks)

The regularizer needs every **pre-ReLU activation**. The obvious PyTorch mechanism is a
`register_forward_hook` / `register_forward_pre_hook`. **We deliberately avoid hooks.**
Instead, model classes are written to **return the pre-activations they produce** as part of
their forward output.

**Why:**

- **Explicit data flow.** The activations a caller can regularize are visible in the function
  signature, not captured by side effect into external state.
- **Type-safety & tooling.** A typed return value is checked by `ty`; a dict populated by a
  hook closure is not.
- **Testability.** A forward pass that returns its activations is a pure-ish function you can
  assert on directly — no hook registration/teardown lifecycle in tests.
- **No lifecycle bugs.** Hooks leak if not removed, fire in surprising orders, and interact
  badly with `torch.compile`, data-parallel wrappers, and module surgery.

**The realized interface** (frozen since phase 1a):

```python
from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True, eq=False)  # identity eq/hash — tensors are not value-comparable
class ForwardOutput:
    """A forward pass plus the pre-ReLU activations the regularizer consumes."""

    logits: Tensor                       # [B, C]
    pre_activations: tuple[Tensor, ...]  # one [M, N_ℓ] per SelectiveReLU site, forward order
    site_names: tuple[str, ...]          # stable ids aligned with pre_activations, e.g. ("relu0", "relu1")


class ReLUMLP(nn.Module):
    def forward(self, x: Tensor) -> ForwardOutput:
        x = x.flatten(1)  # MNIST [B,1,28,28] & CIFAR [B,3,32,32] → [B, input_dim]
        pre: list[Tensor] = []
        for linear, act in zip(self.linears, self.activations, strict=True):
            z = linear(x)
            pre.append(z)  # capture the literal input tensor to this activation site
            x = act(z)     # act is a SelectiveReLU; default mode is plain ReLU
        return ForwardOutput(self.head(x), tuple(pre), self.site_names)
```

`ForwardOutput` is `frozen=True, eq=False`: it is a transport struct, so equality/hashing
fall back to identity and callers compare its tensor fields with `torch.equal`, never `==`
on the whole object. The activation at each site is a **`SelectiveReLU`**, not a bare
`torch.relu`: it carries an int64 `mode` buffer of shape `[N]` (default all-`RELU`) and per
neuron applies `RELU` (plain `torch.relu`), `ZERO` (dead → outputs zeros), or `IDENTITY`
(passthrough → outputs `z`). With every neuron in `RELU` mode its output equals
`torch.relu(z)` bit-for-bit (invariant **I1**), so an untouched model is exactly the ReLU
network it always was. Surgery is performed by setting modes on a `deepcopy` of the model;
the canonical model is never mutated.

The `linears`/`activations` accessors are typed properties over the model's
`nn.ModuleList` containers: `ty` erases the element types of a `ModuleList`, so the
properties re-cast them to `list[nn.Linear]` / `list[SelectiveReLU]` while returning the
same live, registered modules (so callers can write `model.activations[i].set_modes(...)`).
The regularizer consumes `output.pre_activations` directly.

Both models subclass **`PreActModel`** (`models/base.py`) — the
emits-its-own-pre-activations contract that trainer/analysis/surgery signatures accept —
and are built via `build_model(config)` dispatching on `ModelConfig.kind`.

## Key decision: a conv "neuron" is an individual position, not a channel

For **`ReLUCNN`** a "neuron" is an **individual output position** `(c, h, w)`,
statistically identical to an MLP neuron — its `q_i` is sampled over the **batch only**.
Each conv site flattens its pre-activations to `[B, C·H·W]` (index
`i = (c·H + h)·W + w`), applies its `SelectiveReLU(C·H·W)` on that view, and reshapes
back before pooling — elementwise ops keep the flattened path bit-identical to
`relu(conv(x))` (invariant I1), the `[M, N_ℓ]` contract above holds unchanged, and
per-position `ZERO`/`IDENTITY` masking stays bit-exact.

**Why not per-channel** (pooling a channel's positions into the sample dimension): that
statistic *mismeasures* sign-consistency. A channel whose top-half positions are
always-positive and bottom-half always-negative is perfectly sign-consistent at scalar
level — fully eliminable (`IDENTITY` above, `ZERO` below, bit-exact) — yet its pooled
`p ≈ 0.5` earns the maximal entropy penalty, and the gradient pushes the
minority-direction positions to flip their already-consistent sign. Per-position keeps
the "batch" meaning *data samples* (the faithful generalization of the MLP math) and is
the granularity the range-analysis/decompilation phases consume. Channel-level views
stay **derivable** (`q_c` = the mean of its positions' `q`s; channel masking = a uniform
position mask), and a channel-pooled regularizer remains available as an explicit
experimental arm via `RegConfig.granularity="channel"`
(`grouped_sign_consistency_loss`; groups of size 1 reduce exactly to the per-neuron
loss). Empirically the two arms produce congruent aggregates — see the experiment log.

## Module layout

```
src/kill_nonlinearities/
├── __init__.py          # package marker + __version__
├── config.py            # frozen dataclasses (ModelConfig, …, ExperimentConfig)
├── models/              # networks that EMIT pre-activations (no hooks)
│   ├── __init__.py      #   build_model(ModelConfig) factory (kind: mlp | cnn)
│   ├── outputs.py       #   ForwardOutput (frozen, eq=False)
│   ├── activations.py   #   ActivationMode; SelectiveReLU (int64 mode buffer)
│   ├── base.py          #   PreActModel — the pre-activation contract
│   ├── mlp.py           #   ReLUMLP(ModelConfig) → ForwardOutput
│   └── cnn.py           #   ReLUCNN(ModelConfig) → ForwardOutput (neuron = (c,h,w) position)
├── regularization/      # the sign-consistency loss, as pure functions
│   ├── surrogate.py     #   soft_sign(z, tau)
│   ├── entropy.py       #   binary_entropy; batch/hard_fraction_positive; sign_entropy
│   └── loss.py          #   sign_consistency_loss(pre_activations, tau, eps)
├── training/            # training loop, τ-annealing, logging, checkpoints
│   ├── schedule.py      #   TemperatureSchedule; checkpoint_steps
│   ├── logging.py       #   Logger protocol; NullLogger; InMemoryLogger; WandbLogger
│   ├── checkpoint.py    #   save_checkpoint / load_checkpoint
│   └── trainer.py       #   train(...) → TrainResult
├── data/                # dataloaders + probe selection
│   └── datasets.py      #   make_dataloaders; select_probe_neurons; make_probe_batch
├── analysis/            # hard sign statistics + neuron ranking/selection
│   ├── statistics.py    #   collect_pre_activations; neuron_stats; collect_history
│   └── selection.py     #   rank_by_entropy/rank_random; make_k_grid; select_topk; assign_modes
├── surgery/             # masked-activation surgery (no folding/pruning)
│   └── apply.py         #   apply_modes; evaluate_accuracy; k_sweep
├── viz/                 # headless figures + GIFs (matplotlib Agg, imageio+pillow)
│   ├── plots.py
│   └── animation.py
└── experiments/         # end-to-end runner + wandb sweep glue
    ├── run.py           #   run_experiment(config); CLI
    └── sweep.py         #   build_sweep_config; config_from_wandb; sweep_entry; launch_sweep
```

**Surgery is masked-activation only.** Phase 1a does *not* fold linear layers or structurally
prune width; instead it flips each eliminable neuron's `SelectiveReLU` mode to `ZERO` or
`IDENTITY` and measures accuracy-vs-k. Linear-folding/pruning stays out of scope.

### Module responsibilities

| Module | Does | Depends on |
| --- | --- | --- |
| `models/` | define networks; return logits **and** pre-activations; carry `SelectiveReLU` modes | `torch` |
| `regularization/` | turn pre-activations into the scalar $\mathcal{L}_{\text{reg}}$ (loss clamps $p$) | `torch` |
| `training/` | optimize $\mathcal{L}_{\text{task}} + \lambda\mathcal{L}_{\text{reg}}$; anneal $\tau$; log; checkpoint | `models/`, `regularization/` |
| `data/` | build deterministic train/val/test loaders; pick fixed probe neurons/inputs | `torch`, `torchvision` |
| `analysis/` | measure $q_i$, $H(q_i)$; rank neurons; build the k-grid; assign modes | `models/`, `regularization/` |
| `surgery/` | apply modes to a deepcopy; evaluate accuracy; run the k-sweep | `models/`, `analysis/` |
| `viz/` | render static figures + GIFs headlessly (Agg) | `matplotlib`, `imageio`, `pillow` |
| `experiments/` | wire the pipeline end-to-end; wandb logging + sweeps (lazy import) | all of the above, `wandb` |

Each maps directly onto a step in [the method](../research/README.md#the-method). When a
file starts doing more than its row above, that is the signal to split it.

---

[← Architecture](README.md) · [Research / the method →](../research/README.md) · [Development →](../development/README.md)
