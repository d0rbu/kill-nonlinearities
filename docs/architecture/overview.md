# Architecture overview (planned)

[← Architecture](README.md) · [← Documentation hub](../README.md)

> **Status:** design intent, not yet implemented. The package
> (`src/kill_nonlinearities/`) currently contains only a version marker. This document
> describes how we *intend* to grow it so that implementation stays coherent. Treat it as a
> living design that the first implementation PR may refine.

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

**Sketch of the intended interface** (illustrative — *not yet implemented*):

```python
from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class ForwardOutput:
    """A forward pass plus the pre-ReLU activations the regularizer consumes."""

    logits: Tensor
    pre_activations: list[Tensor]  # one [batch, features] tensor per regularized ReLU site


class ReLUMLP(nn.Module):
    def forward(self, x: Tensor) -> ForwardOutput:
        pre: list[Tensor] = []
        for linear in self.linears[:-1]:
            z = linear(x)
            pre.append(z)          # capture the pre-activation explicitly
            x = torch.relu(z)      # forward stays a plain ReLU
        return ForwardOutput(logits=self.linears[-1](x), pre_activations=pre)
```

The regularizer then consumes `output.pre_activations` directly. Returning a structured
object (rather than a bare tuple) keeps call sites readable and lets the shape evolve.

## Planned module layout

```
src/kill_nonlinearities/
├── __init__.py          # package marker + __version__ (exists)
├── models/              # model classes that EMIT pre-activations (no hooks)
│   ├── mlp.py           #   phase 1a: ReLU MLP for MNIST/CIFAR
│   └── transformer.py   #   phase 1b: small transformer LM (FFN blocks)
├── regularization/      # the sign-consistency loss, as pure functions
│   ├── surrogate.py     #   soft_sign(z, tau) = sigmoid(z / tau)
│   ├── entropy.py       #   binary_entropy(p); batch_fraction_positive(...)
│   └── loss.py          #   sign_consistency_loss(pre_activations, tau) -> Tensor
├── training/            # training loop, tau-annealing schedule, lambda config
└── analysis/            # hard fraction-positive q_i, neuron classification, metrics/plots
```

A future **`surgery/`** module (phase 2) will replace/prune/fold eliminable units. It is
intentionally absent until phase 1 analysis justifies it.

### Module responsibilities

| Module | Does | Depends on |
| --- | --- | --- |
| `models/` | define networks; return logits **and** pre-activations | `torch` |
| `regularization/` | turn pre-activations into the scalar $\mathcal{L}_{\text{reg}}$ | `torch` |
| `training/` | optimize $\mathcal{L}_{\text{task}} + \lambda\mathcal{L}_{\text{reg}}$; anneal $\tau$ | `models/`, `regularization/` |
| `analysis/` | measure $q_i$, classify neurons, emit metrics/plots | `models/` |

Each maps directly onto a step in [the method](../research/README.md#the-method). When a
file starts doing more than its row above, that is the signal to split it.

---

[← Architecture](README.md) · [Research / the method →](../research/README.md) · [Development →](../development/README.md)
