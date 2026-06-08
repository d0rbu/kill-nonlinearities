# Research: killing nonlinearities

[← Documentation hub](../README.md) · [← Repository root](../../README.md)

This is the canonical writeup of the research idea and method, followed by the
running **experiment log**. It doubles as the lab notebook: the formal method lives
at the top; what we actually learn from experiments accumulates in
[Experiment log](#experiment-log) below.

> **Status:** method specified; no experiments run yet. This document describes the
> *intended* method — it is a design, not a report of results.

---

## Table of contents

- [Motivation](#motivation)
- [The method](#the-method)
- [Why minimizing entropy enforces sign-consistency](#why-minimizing-entropy-enforces-sign-consistency)
- [Subtleties and failure modes](#subtleties-and-failure-modes)
- [From consistency to elimination](#from-consistency-to-elimination)
- [What we measure (phase 1)](#what-we-measure-phase-1)
- [Terminology](#terminology)
- [Roadmap](#roadmap)
- [Experiment log](#experiment-log)

---

## Motivation

A ReLU, `max(0, z)`, is only *nonlinear* for a neuron when that neuron's pre-activation
`z` is **positive for some inputs and negative for others**. If, across the data
distribution, a neuron's pre-activation always keeps the **same sign**, its ReLU does
nothing interesting:

- **Always positive** (`z > 0` for all inputs) → `ReLU(z) = z` → the unit is effectively
  **linear** (an identity passthrough).
- **Always negative** (`z < 0` for all inputs) → `ReLU(z) = 0` → the unit is **dead**
  (it can be removed).

Only neurons whose sign genuinely *switches* across inputs are doing nonlinear work.

This matters for two reasons:

1. **Interpretability.** Nonlinearities are what make networks hard to reason about. A
   layer whose ReLUs are all sign-consistent is a *linear* map on the data it actually
   sees, and linear maps are analyzable. If we can push most nonlinearity into a small,
   identifiable set of neurons, we localize where the "interesting" computation happens.
2. **Architecture.** Sign-consistent units are removable or collapsible. Replacing
   always-on ReLUs with identities lets consecutive linear layers fold into one; removing
   always-off units shrinks width. The result is a smaller, partially-linearized network
   with (ideally) similar behavior.

The hypothesis: **we can *encourage* a network to concentrate its nonlinearity** by adding
a regularizer that rewards sign-consistent pre-activations, then quantify how much
nonlinearity was actually unnecessary.

## The method

Take one ReLU site (a layer) with pre-activations $z \in \mathbb{R}^{B \times N}$ for a
batch of $B$ inputs and $N$ neurons; $z_{b,i}$ is neuron $i$'s pre-ReLU value for sample
$b$. The construction below applies independently at every regularized ReLU site and is
averaged over them.

**The forward pass is unchanged — it stays a plain ReLU:**

$$a_{b,i} = \mathrm{ReLU}(z_{b,i}) = \max(0,\, z_{b,i}).$$

The regularizer is a **pure side-computation** on the pre-activations $z$. It never alters
what the network computes; it only adds a term to the loss. This means there is **no
train/inference mismatch** — at inference the model is exactly the ReLU network it always
was.

**Step 1 — soft "fired positive" probability (temperature $\tau$).** The hard sign
$\mathrm{sign}(z)$ has zero gradient almost everywhere, so it cannot drive learning.
Replace it, *in the regularizer only*, with a temperature-scaled sigmoid:

$$s_{b,i} = \sigma\!\left(\frac{z_{b,i}}{\tau}\right), \qquad \sigma(u) = \frac{1}{1+e^{-u}}, \qquad \tau > 0.$$

$s_{b,i} \in (0,1)$ is a smooth, differentiable surrogate for "neuron $i$ fired positive on
sample $b$": $s \to 1$ when $z \gg 0$, $s \to 0$ when $z \ll 0$, and $s = \tfrac12$ at
$z = 0$. As $\tau \to 0^+$ it converges to the rescaled hard sign
$\tfrac{1}{2}(\mathrm{sign}(z)+1)$.

**Step 2 — batch-mean per neuron.** Average the surrogate over the batch to get, for each
neuron, the (soft) **fraction of the batch for which it fired positive**:

$$p_i = \frac{1}{B}\sum_{b=1}^{B} s_{b,i} \;\in\; (0,1).$$

$p_i$ is a differentiable Monte-Carlo estimate of $\Pr_{x\sim\text{data}}[\,z_i(x) > 0\,]$
(smoothed by $\tau$).

**Step 3 — binary entropy per neuron.**

$$H(p_i) = -\,p_i \log p_i - (1-p_i)\log(1-p_i).$$

$H$ is maximal at $p_i = \tfrac12$ (the neuron flips sign half the time — maximally
nonlinear) and approaches zero as $p_i \to 0$ or $1$ (perfect sign-consistency).

**Step 4 — the regularization loss.** Average the entropy over all $N$ regularized neurons
(and over all regularized layers $\ell$):

$$\mathcal{L}_{\text{reg}} = \frac{1}{|\mathcal{L}|}\sum_{\ell \in \mathcal{L}} \frac{1}{N_\ell}\sum_{i=1}^{N_\ell} H\!\big(p_i^{(\ell)}\big).$$

**Step 5 — the total objective.**

$$\mathcal{L} = \mathcal{L}_{\text{task}} + \lambda\, \mathcal{L}_{\text{reg}}, \qquad \lambda \ge 0.$$

We **minimize** $\mathcal{L}$. Minimizing $\mathcal{L}_{\text{reg}}$ drives every $p_i$
toward $0$ or $1$, i.e. toward sign-consistency. $\lambda$ is the central knob trading task
performance against how aggressively nonlinearities are removed.

**Temperature annealing.** Start with a larger $\tau$ (smooth surrogate, informative
gradients everywhere) and **anneal $\tau$ downward** over training, so the surrogate
sharpens toward the true sign as the consistent neurons emerge. $\tau$ and its schedule are
hyperparameters; see [Subtleties](#subtleties-and-failure-modes).

## Why minimizing entropy enforces sign-consistency

The *hard* statistic $q_i$ (and the soft $p_i$ in the $\tau \to 0$ limit) equals $0$ or $1$
**exactly** iff every sample in the batch lands on the same side of zero. At finite $\tau$ the
soft $p_i$ stays strictly inside $(0,1)$, but minimizing $H(p_i)$ drives it arbitrarily close
to an endpoint — and as $\tau$ anneals, "near an endpoint" becomes "sign-consistent." Any
mixture of signs holds $p_i$ away from the extremes at a positive entropy cost, so the
gradient pressure is directly toward "all-positive" or "all-negative" per neuron.

The gradient of the entropy w.r.t. $p$ (natural log) is

$$\frac{\mathrm{d}H}{\mathrm{d}p} = \log\!\frac{1-p}{p},$$

which is positive for $p < \tfrac12$ and negative for $p > \tfrac12$. Gradient descent on
$H$ therefore pushes $p$ **away from $\tfrac12$ toward whichever extreme it is already
nearer** — the batch "commits" to the side it currently leans toward. Note $p = \tfrac12$
is an *unstable* equilibrium with zero entropy gradient (see failure modes).

## Subtleties and failure modes

- **Soft/hard gap.** The regularizer optimizes the *soft* $p_i$, but elimination later uses
  the *hard* sign. The two agree only as $\tau \to 0$. Anneal $\tau$ so that, by the end of
  training, "soft sign-consistent" implies "hard sign-consistent." Validate with the hard
  statistic (next section), never the soft one.
- **The $p = \tfrac12$ unstable equilibrium.** A neuron whose batch is perfectly balanced gets no entropy
  gradient. In practice the task loss, mini-batch noise, and asymmetric initialization break
  the tie; a perfectly balanced average is measure-zero. Worth watching, not engineering
  around prematurely.
- **$\lambda$ too large → linear collapse.** Push hard enough and the network linearizes
  itself into the ground, destroying capacity and task accuracy. Too small and nothing
  happens. The interesting result is the **trade-off curve** (accuracy vs. fraction of
  neurons made consistent) as $\lambda$ sweeps.
- **Batch-size noise.** $p_i$ from a single mini-batch is a noisy estimate of the population
  sign-probability. A documented variant: maintain an **exponential moving average** of
  $p_i$ across batches for a lower-variance population estimate before applying entropy.
- **Temperature trade-off.** Large $\tau$: smooth, well-behaved gradients but a loose proxy
  for the sign. Small $\tau$: faithful to the sign but gradients vanish away from $z=0$.
  Annealing navigates between the two.
- **Biases and normalization.** The sign of $z$ depends on the learned bias (and on any
  BatchNorm/LayerNorm before the ReLU). The regularizer can satisfy itself by shifting biases
  to make signs consistent — usually the desired behavior, but something to keep in mind when
  interpreting results.
- **Alternative formulations (future variants).** Entropy of the batch-mean is the
  formulation specified here. Alternatives worth comparing: penalizing per-sample confidence
  $|s_{b,i} - \tfrac12|$, penalizing the variance of the sign across the batch, or a
  straight-through estimator on the hard sign. These are noted for later, not committed to.

## From consistency to elimination

> **Scope note.** The near-term work is **analysis-first** (phase 1): implement the
> regularizer and *measure* how sign-consistent the network becomes and at what cost. The
> actual network surgery below is **phase 2 (future)** — described here so the end-to-end
> story is on record. See the [roadmap](#roadmap).

After training, evaluate the **hard** fraction-positive of each neuron over a held-out set
of $M$ inputs:

$$q_i = \frac{1}{M}\sum_{m=1}^{M} \mathbb{1}[\,z_i(x_m) > 0\,].$$

Pick a tolerance $\varepsilon$ and classify:

| Condition | Meaning | Action (phase 2) |
| --- | --- | --- |
| $q_i \ge 1-\varepsilon$ | consistently **positive** (ReLU ≈ identity) | replace ReLU with an identity **passthrough** → unit becomes linear |
| $q_i \le \varepsilon$ | consistently **negative** (ReLU ≈ 0) | unit is **dead** → prune it |
| otherwise | genuinely nonlinear | keep the ReLU |

Once units are linearized, consecutive linear maps can be **folded**: for adjacent linear
layers, $W_2(W_1 x + b_1) + b_2 = (W_2 W_1)x + (W_2 b_1 + b_2)$. Folding plus pruning yields
a smaller, partially-linear network. The phase-2 deliverable is to perform this surgery and
measure how much accuracy is retained.

## What we measure (phase 1)

- **Consistency.** Fraction of neurons that are *eliminable* — $q_i$ within $\varepsilon$ of
  $0$ or $1$ — as a function of $\lambda$.
- **Trade-off.** Task accuracy vs. $\lambda$, and accuracy vs. fraction eliminable. This
  curve is the headline result.
- **Distribution of $q_i$.** A histogram of hard fraction-positive across neurons; we expect
  it to grow **bimodal** (mass piling at $0$ and $1$) as $\lambda$ increases.
- **Regularizer trajectory.** $\mathcal{L}_{\text{reg}}$ and the chosen $\tau$ schedule over
  training.

## Terminology

| Term | Definition |
| --- | --- |
| **Pre-activation** $z$ | the input to a ReLU (post-linear, pre-nonlinearity). |
| **Sign-consistent neuron** | a neuron whose pre-activation keeps the same sign across (nearly) all inputs. |
| **Soft fraction positive** $p_i$ | $\frac1B\sum_b \sigma(z_{b,i}/\tau)$ — differentiable, used in the loss. |
| **Hard fraction positive** $q_i$ | $\frac1M\sum_m \mathbb{1}[z_i>0]$ — the true statistic, used to classify/eliminate. |
| **Temperature** $\tau$ | sigmoid scale; small $\tau$ → sharper approximation of the sign. |
| **Strength** $\lambda$ | weight on $\mathcal{L}_{\text{reg}}$ in the total loss. |
| **Eliminable neuron** | a neuron whose ReLU can be replaced (passthrough) or removed (prune). |
| **Passthrough** | replacing an always-on ReLU with the identity, linearizing the unit. |

## Roadmap

Tracked work, roughly in order. (Once a GitHub remote exists, these become issues — see
[contributing](../development/contributing.md).)

- [x] **Phase 0 — scaffold.** Tooling, test harness, docs (this repo, now).
- [ ] **Phase 1a — regularizer + analysis on a toy MLP** (MNIST/CIFAR). Implement
  $\mathcal{L}_{\text{reg}}$, sweep $\lambda$, produce the trade-off curve and $q_i$
  histograms. Models emit their own pre-activations (see
  [architecture](../architecture/overview.md)).
- [ ] **Phase 1b — small transformer (language modeling).** Apply the same regularizer to
  the MLP/FFN blocks of a small transformer.
- [ ] **Phase 2 — surgery.** Replace/prune/fold eliminable units and measure retained
  accuracy.

## Experiment log

> Newest entries on top. Each entry: date, what was tried, config (λ, τ schedule, model,
> data), result, and takeaway. Keep findings here so knowledge accumulates in one place.

_No experiments run yet._

```
### YYYY-MM-DD — <one-line title>
- **Setup:** model / data / λ / τ schedule / seed
- **Result:** numbers, plots (link artifacts)
- **Takeaway:** what we now believe, and what to try next
```

---

[← Documentation hub](../README.md) · [Development docs →](../development/README.md) · [Architecture →](../architecture/overview.md)
