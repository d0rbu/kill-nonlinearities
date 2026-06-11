# Convolutional ReLU sites (`ReLUCNN`) — design

**Date:** 2026-06-10 · **Status:** implemented · **Extends:**
[phase-1a spec](2026-06-08-phase1a-regularizer-and-surgery-design.md)

Adds a convolutional model to phase 1a so the regularizer/surgery pipeline can run
on an architecture that actually fits CIFAR-10 (the flat MLP is ceiling-limited at
~52%; see the [experiment log](../research/README.md#experiment-log)). The pipeline
(regularizer, analysis, surgery, viz, sweep) is reused **unchanged** — the only new
code is the model, its config surface, and the model factory.

## The one real decision: a conv "neuron" is an **individual position** `(c, h, w)`

At a conv ReLU site with pre-activations `z ∈ [B, C, H, W]`, the per-neuron
statistics (`p_i`, `q_i`, `H`) are computed **per scalar output position**:
neuron `i = (c, h, w)`, sampled over the **batch only** —
`q_{c,h,w} = Pr_x[z_{c,h,w}(x) > 0]`. This is the faithful generalization of the
MLP math: the "batch" stays *samples from the data distribution*, and a conv
layer is just a weight-tied linear layer whose scalar outputs are the `(c, h, w)`
units.

- **Why not per-channel (rejected first draft):** pooling positions into the
  sample dimension (`q_c` over all `(b, h, w)`) *mismeasures* sign-consistency.
  Counterexample: a channel whose top-half positions are always-positive and
  bottom-half always-negative is perfectly sign-consistent at scalar level —
  fully eliminable (`IDENTITY` above, `ZERO` below, bit-exact) — yet its pooled
  `p_c ≈ 0.5` earns the *maximal* entropy penalty, and the gradient pushes the
  minority-direction positions to flip their already-consistent sign. The pooled
  statistic conflates "switches sign across inputs" (true nonlinearity) with
  "varies across positions" (harmless; still linear per unit), and incentivizes
  whole channels to saturate all-on/all-off.
- **Losslessness is per-position**, exactly as in the phase-1a spec: `q_i = 0`
  → `ZERO` is bit-exact on the selection set; `q_i = 1` (strict `z > 0`) →
  `IDENTITY` is bit-exact.
- **Channel-level views are derivable, not primitive.** `q_c` is the mean of its
  positions' `q`s, and channel masking is a uniform position mask — so
  channel-granular analyses (and conv-preserving folds, which need whole-`IDENTITY`
  channels) can be recovered from position-level results later.
- **Channel-pooled regularizer variant (addendum, same day).** For comparison,
  `RegConfig.granularity = "channel"` trains with
  `grouped_sign_consistency_loss`, which pools each conv channel's positions
  into the sample dimension before the entropy (`PreActModel.site_group_sizes`
  carries the per-site group width; groups of 1 at MLP/fc sites make it equal
  the per-neuron loss there). **Only the training incentive changes** —
  analysis and surgery stay per-position, so both arms are measured on the same
  axis. This is exactly the rejected-default semantics above, kept as an
  explicit experimental arm (config:
  [`configs/cifar10-cnn-chanreg.json`](../../configs/cifar10-cnn-chanreg.json)).
- **Alignment with the roadmap:** phase 4's range analysis bounds each scalar
  pre-activation and phase 5's decompilation branches on each scalar ReLU —
  position granularity is what those phases consume.

## Mechanics (no new math, no new consumers)

1. **Flattened application.** `SelectiveReLU.mode` is `[N]` and broadcasts over
   the last dim. Conv sites compute `z = conv(x)` (`[B, C, H, W]`), flatten to
   `z_flat = z.flatten(1)` (`[B, C·H·W]`, index `i = (c·H + h)·W + w`), apply the
   site's `SelectiveReLU(C·H·W)` there, and reshape back before pooling.
   `torch.relu`/`torch.where` are elementwise, so the flattened path is
   **bit-identical** to plain `relu(conv(x))` (invariant I1 holds; there is a
   test).
2. **2-D emission.** Each conv site emits `pre_activations[i] = z.flatten(1)` —
   shape `[B, C·H·W]`. Every downstream consumer (`sign_consistency_loss`,
   `collect_pre_activations`, `neuron_stats`, probe GIFs) already treats dim 0 as
   "samples" and dim 1 as "neurons", so the contract *"one `[M, N]` tensor per
   site"* is unchanged. FC sites emit `[B, N]` exactly as in `ReLUMLP`. Note the
   neuron counts are large (e.g. `32·32·32 = 32,768` for a 32-channel site on
   CIFAR) — per-site *mean* entropy keeps the loss scale unchanged, and the
   k-grid/ranking are O(total neurons), which stays cheap.
3. **Architecture.** `(Conv2d(3×3, padding 1) → SelectiveReLU → MaxPool2d(2))*`
   over `conv_channels`, then flatten → `(Linear → SelectiveReLU)*` over
   `hidden_dims` → `Linear` head. Site names: `conv0..convN-1`, then `fc0..`.
   Kernel/stride/pool are fixed (3/1/2) — knobs can be added when an experiment
   needs them, not before.
4. **Factory + shared base.** `models/base.py` defines `PreActModel` (the
   "emits-its-own-pre-activations" contract: `site_names`, `activations`,
   `forward → ForwardOutput`); `ReLUMLP` and `ReLUCNN` subclass it, and
   trainer/analysis/surgery signatures take `PreActModel`. `build_model(config)`
   dispatches on `ModelConfig.kind`.

## Config surface

`ModelConfig` gains `kind: "mlp" | "cnn"` (default `"mlp"` — all existing configs
unchanged), `in_channels` (default 3), `image_size` (default 32), and
`conv_channels` (default `()`). Validation: `cnn` requires non-empty
`conv_channels` and `image_size` divisible by `2^len(conv_channels)` (one 2×2
pool per block). `input_dim` is ignored for `cnn` (the head's fan-in is derived).

## Ripple changes

- `make_probe_batch` no longer flattens inputs — it returns the dataset's native
  shape (`ReLUMLP` flattens internally anyway; `ReLUCNN` needs `[B, C, H, W]`).
- The synthetic provider emits `[size, in_channels, image_size, image_size]`
  when `model.kind == "cnn"` (flat vectors otherwise), so the offline
  integration tests cover the CNN end-to-end.

## Non-goals (deferred)

Linear folding of `IDENTITY` units (phase 2); stride/kernel/pool/BatchNorm
options; a channel-pooled regularizer variant (rejected as the default above;
derivable post-hoc for analysis).
