# Research: killing nonlinearities

[← Documentation hub](../README.md) · [← Repository root](../../README.md)

This is the canonical writeup of the research idea and method, followed by the
running **experiment log**. It doubles as the lab notebook: the formal method lives
at the top; what we actually learn from experiments accumulates in
[Experiment log](#experiment-log) below.

> **Status:** Phase 1a **in progress** — the regularizer, training, analysis, and
> masked-activation surgery are implemented (see the
> [phase-1a spec](../specs/2026-06-08-phase1a-regularizer-and-surgery-design.md)). Real
> MNIST/CIFAR results land in the [experiment log](#experiment-log) as runs complete.

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

> **Scope note.** Phase 1a is **analysis-first plus masked-activation surgery**: we implement
> the regularizer, *measure* sign-consistency and its cost, and perform surgery by flipping a
> neuron's activation **mode** (not by folding linear layers). **Linear-folding / structural
> pruning remains a later phase** — see the [roadmap](#roadmap).

After training, evaluate the **hard** fraction-positive of each neuron over a held-out set
of $M$ inputs:

$$q_i = \frac{1}{M}\sum_{m=1}^{M} \mathbb{1}[\,z_i(x_m) > 0\,].$$

Pick a tolerance $\varepsilon$ and classify:

| Condition | Meaning | Action (masked surgery) |
| --- | --- | --- |
| $q_i \ge 1-\varepsilon$ | consistently **positive** (ReLU ≈ identity) | set mode to `IDENTITY` → passthrough, unit becomes linear |
| $q_i \le \varepsilon$ | consistently **negative** (ReLU ≈ 0) | set mode to `ZERO` → unit is **dead** (outputs 0) |
| otherwise | genuinely nonlinear | keep `RELU` |

**Masked-activation surgery (what phase 1a does).** Rather than rewriting weights, we set a
per-neuron **mode** on a `SelectiveReLU`: an always-negative neuron ($q_i \le \varepsilon$) is
set to `ZERO` (its output is forced to 0, exactly as its ReLU already produced), and an
always-positive neuron ($q_i \ge 1-\varepsilon$) is set to `IDENTITY` (passthrough, linearizing
the unit). Genuinely nonlinear neurons keep `RELU`. The forward pass and weights are otherwise
untouched, so this is a clean, reversible measurement of *how much* nonlinearity was
unnecessary. **Linear folding** — collapsing $W_2(W_1 x + b_1) + b_2 = (W_2 W_1)x + (W_2 b_1 +
b_2)$ across adjacent linearized layers — and structural width pruning are deferred to a later
phase; phase 1a measures the accuracy-vs-k trade-off of masking alone.

### Selection-set-only losslessness

Masking a neuron is **exactly lossless only when its hard sign-entropy is truly zero** — i.e.
$q_i$ is **exactly** 0 or 1 (using the strict $z > 0$ predicate) on the set it was measured on.
A $q=0$ neuron set to `ZERO` is lossless even if some $z = 0$ in the batch (because
$\mathrm{ReLU}(0) = 0$); a $q=1$ neuron set to `IDENTITY` requires **all** $z > 0$ strictly. For
selected neurons with *interior* $q$ (chosen by how far down the ascending-entropy ranking we
cut), masking changes the logits, and the **accuracy-vs-k** curve measures that loss by design.
Crucially the guarantee is **on the selection set only**: held-out (test) inputs may push a
"consistent" neuron across zero, which is exactly why we report the val-vs-test accuracy-vs-k
gap.

### Gradient safety: the loss clamps p

The entropy's forward value is exactly 0 at $p \in \{0, 1\}$, but its derivative
$\mathrm{d}H/\mathrm{d}p = \log\frac{1-p}{p}$ diverges there. As $\tau$ anneals low, the soft
$p_i$ of a sign-consistent neuron saturates to *exactly* 0 or 1 in float32, so a naive
`loss.backward()` would inject NaN — destroying the model precisely when it succeeds.
Therefore the **loss** clamps $p$ into $[\varepsilon, 1-\varepsilon]$ before computing entropy
(clamp's backward is 0 outside the range, so saturated neurons get a finite zero gradient).
The `binary_entropy` function itself stays **unclamped** — the honest math — and is used by
**analysis** on the hard $q$ under `no_grad`, where the endpoint gradient never arises.

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

Tracked work, roughly in order. (File these as GitHub issues — see
[contributing](../development/contributing.md#roadmap--issues).)

- [x] **Phase 0 — scaffold.** Tooling, test harness, docs (this repo).
- [~] **Phase 1a — regularizer + analysis + masked surgery on a toy MLP** (MNIST/CIFAR).
  *In progress.* $\mathcal{L}_{\text{reg}}$, τ-annealing, λ-sweep via wandb Sweeps, the
  trade-off curve, $q_i$ histograms, and masked-activation surgery (accuracy-vs-k on val+test)
  are implemented per the [phase-1a spec](../specs/2026-06-08-phase1a-regularizer-and-surgery-design.md).
  Models emit their own pre-activations (see [architecture](../architecture/overview.md)).
- [ ] **Phase 1b — small transformer (language modeling).** Apply the same regularizer to the
  MLP/FFN blocks of a small transformer.
- [ ] **Phase 2 — structural surgery.** Fold linearized layers and prune dead units; measure
  retained accuracy. (Phase 1a already does the *masked* form.)

## Experiment log

> Newest entries on top. Each entry: date, what was tried, config (λ, τ schedule, model,
> data), result, and takeaway. Keep findings here so knowledge accumulates in one place.

### 2026-06-08 — MNIST λ>0 baseline (regularizer + masked surgery)
- **Setup:** ReLUMLP (784→256→256→10, 512 hidden neurons) / MNIST / λ=0.05 / τ exponential
  1.0→0.1 / Adam lr 1e-3 / 8 epochs / seed 0 / CPU. Config:
  [`configs/mnist.json`](../../configs/mnist.json). Run offline with `mode="disabled"`:
  `uv run python -m kill_nonlinearities.experiments.run --config configs/mnist.json`.
- **Result:** val acc **0.9778**, test acc **0.9791** (k=0, the untouched model). The
  regularizer drove the mean per-site sign-entropy down over training (final
  $\mathcal{L}_{\text{reg}}=0.392$). Of the 512 neurons, **51 reached exact $q=0$** (all dead;
  none hit $q=1$ at this λ) — the lossless-prefix marker — and **86 had sign-entropy < 0.05
  nats**. Masking those 51 via `ZERO` is **bit-exactly lossless on the selection set**: the
  acc-vs-k curve is flat (val 0.9778, test 0.9791) through k=51 and stays ≥ 0.977 through
  k≈128, then degrades smoothly (k=205: val 0.974; k=256: val 0.9522; all-512-masked collapses
  to ~0.30, chance-ish). The **entropy ranking dominates the random baseline** across the
  mid-range (k=256: 0.9522 vs 0.8162 random; k=179: 0.9775 vs 0.9440), confirming low-entropy
  neurons really are the cheap-to-remove ones. Val and test curves track within ~1% everywhere
  (small generalization gap). Artifacts in `runs/mnist-lambda/`: `acc_vs_k.png`,
  `entropy_map.png`, `mean_pre_dist.png`, `per_layer_entropy.png`, `soft_vs_hard.png`,
  `loss_curves.png`, `activation.gif`, `qi_bimodality.gif` (one frame per checkpoint, 9 total).
- **Takeaway:** at λ=0.05 about **10% of neurons (51/512) are exactly removable with zero
  accuracy cost**, and ~25–40% can be masked for a few points of accuracy — the entropy
  ranking is the right knob (it beats random masking by a wide margin mid-curve). At this λ the
  pressure produced dead ($q=0$) units rather than passthrough ($q=1$) ones. Next: sweep λ to
  push more neurons to the endpoints and grow the bimodal $q_i$ split, and run the deeper
  CIFAR-10 MLP.

### 2026-06-08 — CIFAR-10 λ>0 (regularizer + masked surgery) — *queued*
- **Setup:** ReLUMLP (3072→256→256→10) / CIFAR-10 / λ=0.05 / τ exponential 1.0→0.1 / Adam
  lr 1e-3 / 10 epochs / seed 0 / CPU. Config:
  [`configs/cifar10.json`](../../configs/cifar10.json).
- **Result:** *not yet run.* Reproduce it (downloads CIFAR-10 to `./data`, trains on CPU) with:
  ```bash
  uv run python -m kill_nonlinearities.experiments.run --config configs/cifar10.json
  ```
  For an online run with wandb tracking, set `"mode": "online"` in `configs/cifar10.json` and
  `uv run wandb login` first; artifacts land under `runs/cifar10-lambda/` and stream to wandb.
  To produce the λ trade-off curve, launch a wandb sweep over `{lr, epochs, batch_size, λ}`
  from your own wandb account:
  ```bash
  uv run python -c "from kill_nonlinearities.experiments.sweep import launch_sweep; \
  print(launch_sweep({'lr':[1e-3,3e-3],'epochs':[10,20],'batch_size':[64,128],'lam':[0.0,0.01,0.1]}, method='grid', count=12))"
  ```
- **Takeaway:** *pending the run.* CIFAR-10 on a flat MLP is a harder, lower-ceiling task than
  MNIST, so expect fewer exactly-consistent neurons at the same λ; the headline comparison is
  the acc-vs-k val/test gap against MNIST.

```
### YYYY-MM-DD — <one-line title>
- **Setup:** model / data / λ / τ schedule / seed
- **Result:** numbers, plots (link artifacts)
- **Takeaway:** what we now believe, and what to try next
```

---

[← Documentation hub](../README.md) · [Development docs →](../development/README.md) · [Architecture →](../architecture/overview.md)
