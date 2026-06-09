# Spec — Phase 1a: regularizer, training, analysis & surgery

**Date:** 2026-06-08 · **Status:** rev 2 (post adversarial review) · **Owner:** d0rbu
**Implements:** roadmap [Phase 1a](../research/README.md#roadmap) + the masked-activation form of
[Phase 2 surgery](../research/README.md#from-consistency-to-elimination).

This is the design of record for the first implementation milestone. It builds on the method in
[`docs/research/README.md`](../research/README.md) and the architecture decisions in
[`docs/architecture/overview.md`](../architecture/overview.md). Rev 2 folds in a full adversarial
review (4 blockers + majors/minors); fixes are marked `[Rxx]` where useful. The implementation plan
(task list) is produced separately after this spec is approved.

---

## 1. Goal & scope

One coherent, fully-tested pipeline:

```
train + regularize → log/checkpoint → analyze (hard sign stats) → surgery k-sweep → plots/GIFs → wandb sweep
```

**In scope:** ReLU MLP that emits its own pre-activations (no hooks), dataset-agnostic; the
sign-consistency regularizer; training with τ-annealing, λ weighting, periodic checkpoints, wandb
logging; analysis (per-neuron hard fraction-positive `q_i`, sign-entropy `H(q_i)`, mean pre-activation,
soft-vs-hard agreement); **surgery as a per-neuron masked activation** (`SelectiveReLU`: top-k by
ascending sign-entropy → `ZERO` (dead) or `IDENTITY` (passthrough)); accuracy-vs-k on val and test;
the visualizations in §5; hyperparameter sweeps over `{lr, epochs, batch_size, λ}` via **wandb
Sweeps**; MNIST (primary) + CIFAR-10, plus a synthetic provider for tests.

**Out of scope (later):** transformers (1b); structural pruning / linear-folding (we do masked-activation
surgery only); distributed/multi-GPU. Default device CPU; CUDA allowed but untested here.

**Success criteria**
1. `just check` green; coverage ≥ 90% (target ~100% on `regularization/`, `models/`, `analysis/`, `surgery/`).
2. Invariants **I1, I2, I3** (§6) hold in tests.
3. End-to-end MNIST run (with **λ>0**) produces every artifact in §5 and logs to wandb.
4. The wandb sweep launches over `{lr, epochs, batch_size, λ}` and aggregates runs. *(This live step is a
   manual online check, not part of the offline test suite — `[R23]`.)*
5. **Reproducibility:** two runs with an **identical config (same `seed`, `batch_size`, `epochs`,
   `num_workers`) on CPU, in one process** produce bit-identical `history` losses and `q_i` (`torch.equal`).
   This does **not** apply across sweep points that vary `batch_size`/`epochs` `[R8]`.

---

## 2. Module layout

Extends [`docs/architecture/overview.md`](../architecture/overview.md). New dirs: `data/`, `viz/`,
`experiments/`, `surgery/`.

```
src/kill_nonlinearities/
├── config.py            # frozen dataclasses (§3)
├── models/
│   ├── outputs.py       # ForwardOutput (eq=False)
│   ├── activations.py   # ActivationMode; SelectiveReLU
│   └── mlp.py           # ReLUMLP(ModelConfig) → ForwardOutput
├── regularization/
│   ├── surrogate.py     # soft_sign(z, tau)
│   ├── entropy.py       # binary_entropy; batch_fraction_positive; hard_fraction_positive; sign_entropy
│   └── loss.py          # sign_consistency_loss(pre_activations, tau, eps)
├── training/
│   ├── schedule.py      # TemperatureSchedule, checkpoint_steps
│   ├── logging.py       # Logger protocol; NullLogger; InMemoryLogger; WandbLogger
│   ├── checkpoint.py    # save_checkpoint / load_checkpoint
│   └── trainer.py       # train(...) → TrainResult
├── data/
│   └── datasets.py      # dataloaders (mnist/cifar10/synthetic); probe selection
├── analysis/
│   ├── statistics.py    # collect_pre_activations; neuron_stats; collect_history
│   └── selection.py     # rank_by_entropy/rank_random; make_k_grid; select_topk; assign_modes
├── surgery/
│   └── apply.py         # apply_modes; evaluate_accuracy; k_sweep
├── viz/
│   ├── plots.py         # static figures (matplotlib, Agg)
│   └── animation.py     # GIF builder (imageio + pillow)
└── experiments/
    ├── run.py           # run_experiment(config); CLI
    └── sweep.py         # build_sweep_config; config_from_wandb; sweep_entry; launch_sweep
```

**Dependencies `[R10]`:** add `torchvision`, `wandb`, `matplotlib`, `imageio`, `pillow` to
**`[project].dependencies`** (runtime — the integration tests import them end-to-end, so they must not be
dev-only). `numpy` arrives transitively. Added via `uv add`.

---

## 3. Configuration (`config.py`)

Frozen dataclasses (`ty`-checked, no new dep). All have defaults; experiments override.

| Dataclass | Fields |
| --- | --- |
| `ModelConfig` | `input_dim:int`, `hidden_dims:tuple[int,...]=(256,256)`, `output_dim:int=10` |
| `OptimConfig` | `lr:float=1e-3`, `weight_decay:float=0.0`, `name:str="adam"` |
| `TempScheduleConfig` | `kind:str="exponential"`(`exponential\|linear\|constant`), `tau_start:float=1.0`, `tau_end:float=0.1`; **validated `tau_start>0` and `tau_end>0` strictly** `[R16]` |
| `RegConfig` | `lam:float=0.0` (λ; see note), `entropy_eps:float=1e-6` (loss-path clamp, §4.4) `[R1]` |
| `DataConfig` | `dataset:str="mnist"`(`mnist\|cifar10\|synthetic`), `batch_size:int=128`, `eval_batch_size:int=512`, `val_fraction:float=0.1`, `split_seed:int=0`, `drop_last:bool=False`, `data_dir:str="data"`, `num_workers:int=0` `[R8][R25]` |
| `ProbeConfig` | `num_neurons:int=16`, `seed:int=0`, `batch_size:int=512` (invariant to `DataConfig.batch_size` sweeps `[R25]`) |
| `CheckpointConfig` | `every_epochs:int=1`, `dir:str="runs"` (always also save step 0 and final) |
| `SurgeryConfig` | `num_k:int=21` (target grid points; realized count = unique, §4.11), `random_baseline:bool=true`, `tie_break:str="identity"` (mode for measure-zero `q=0.5`) |
| `WandbConfig` | `project:str="kill-nonlinearities"`, `entity:str\|None=None`, `mode:str="online"`(`online\|offline\|disabled`), `group:str\|None=None`, `tags:tuple[str,...]=()` |
| `TrainConfig` | `epochs:int=20`, `seed:int=0`, `device:str="cpu"`, `grad_clip:float\|None=None` |
| `ExperimentConfig` | composes all + `name:str` |

> **`RegConfig.lam` default `[R24]`:** the standalone default run is the **λ=0 baseline** (no
> sign-consistency pressure), so the bimodality/lossless-prefix analyses are vacuous on it *by design*.
> The interesting analyses require **λ>0**; the documented MNIST/CIFAR runs (milestone 9) and the sweep
> use λ>0.

`input_dim`/`output_dim` are dataset-derived in the entrypoint (MNIST 784/10, CIFAR 3072/10) and explicit
for the synthetic path.

---

## 4. Components & interfaces

Notation: `z` = pre-activations `[B,N]`, `τ` temperature, `p_i` soft fraction-positive, `q_i` hard
fraction-positive, `λ` strength.

### 4.1 `models/outputs.py`
```python
@dataclass(frozen=True, eq=False)          # [R4] eq=False → identity eq/hash; tensors are not value-comparable
class ForwardOutput:
    logits: Tensor                          # [B, C]
    pre_activations: tuple[Tensor, ...]     # one [B, N_ℓ] per SelectiveReLU site, forward order
    site_names: tuple[str, ...]             # stable ids aligned with pre_activations, e.g. ("relu0","relu1")
```
Transport struct only; tests compare fields with `torch.equal`, never `==` on the whole object `[R4]`.

### 4.2 `models/activations.py`
```python
class ActivationMode(IntEnum):
    RELU = 0; ZERO = 1; IDENTITY = 2

class SelectiveReLU(nn.Module):
    # register_buffer("mode", torch.zeros(num_features, dtype=torch.int64))  # default RELU
    def forward(self, z: Tensor) -> Tensor:
        relu = torch.relu(z)                                        # z==0 → 0 (relu's exact convention)
        out = torch.where(self.mode == int(ActivationMode.RELU), relu, z)            # [R27] int(...)
        return torch.where(self.mode == int(ActivationMode.ZERO), torch.zeros_like(z), out)
    def set_modes(self, modes: Tensor) -> None: ...   # require modes.shape==(num_features,), dtype int64, values∈{0,1,2}
    def reset(self) -> None: ...                       # all RELU
```
- `mode` is an **int64 buffer** `[N]`: in `state_dict`, moves with `.to(device)`, never trained `[R27]`.
- Broadcasts `[N]` over `[B,N]`. **All-`RELU` ⇒ output is exactly `torch.relu(z)`** (`torch.equal`) — invariant **I1**.
- **Gradient routing** (tested by exact analytic assertions, not gradcheck `[R12]`): `RELU` → `(z>0)` (relu's
  subgradient, `z==0`→0); `IDENTITY` → 1; `ZERO` → 0 (output independent of `z`).

### 4.3 `models/mlp.py`
```python
class ReLUMLP(nn.Module):
    # linears: ModuleList[Linear]; activations: ModuleList[SelectiveReLU]; head: Linear; site_names: tuple
    def forward(self, x: Tensor) -> ForwardOutput:
        x = x.flatten(1)                                   # MNIST [B,1,28,28] & CIFAR [B,3,32,32] → [B, input_dim]
        pre = []
        for linear, act in zip(self.linears, self.activations, strict=True):
            z = linear(x); pre.append(z); x = act(z)
        return ForwardOutput(self.head(x), tuple(pre), self.site_names)
```
- `pre_activations[ℓ]` **is the literal input tensor to `activations[ℓ]`** — the exact value analysis and
  surgery reason about (load-bearing for I2 bit-exactness `[R7]`).
- `input_dim` is owned by the entrypoint (dataset-derived / explicit for synthetic).

### 4.4 `regularization/`
```python
# surrogate.py
def soft_sign(z: Tensor, tau: float) -> Tensor:          # sigmoid(z/tau); raise ValueError if tau <= 0  [R16]
# entropy.py  — natural log (nats); xlogy so the FORWARD value is exact 0 at p∈{0,1}
def binary_entropy(p: Tensor) -> Tensor:                 # -(xlogy(p,p)+xlogy(1-p,1-p)); see gradient note
def batch_fraction_positive(z: Tensor, tau: float) -> Tensor:    # soft_sign(z,tau).mean(0) → [N]  (p_i); requires B>=1
def hard_fraction_positive(z: Tensor) -> Tensor:         # (z > 0).to(z.dtype).mean(0) → [N]  (q_i); strict '>'  [R6]
def sign_entropy(q: Tensor) -> Tensor:                   # binary_entropy(q); used on hard q (no-grad analysis)
# loss.py
def sign_consistency_loss(pre_activations: Sequence[Tensor], tau: float, eps: float) -> Tensor:
    # per site: p = batch_fraction_positive(z, tau); p = p.clamp(eps, 1-eps);  H = binary_entropy(p).mean()
    # return mean over sites of H   → scalar
```

> **Gradient safety — BLOCKER fix `[R1]`.** `binary_entropy` is exact in the forward at `p∈{0,1}` but its
> backward is `dH/dp = log((1-p)/p) → ±∞` (NaN in autograd). With `τ_end=0.1`, `sigmoid(z/τ)` saturates to
> **exactly** 1.0/0.0 in float32 for `|z|≳1.7`, so a sign-consistent neuron yields `p_i∈{0,1}` and
> `loss.backward()` injects NaN — the regularizer destroys the model **exactly when it succeeds**. Therefore
> the **loss clamps `p` into `[eps, 1-eps]`** (`RegConfig.entropy_eps`, default 1e-6) before `binary_entropy`.
> Clamp's backward is 0 outside the range, so fully-saturated neurons get a **finite (zero) gradient** — safe,
> and correct (they need no further push). `binary_entropy` itself stays unclamped (the honest math) and is
> used by **analysis** on the hard `q` under `no_grad`, where endpoint NaN gradients never arise. This does
> **not** weaken **I2**, which is measured on the hard `q` path.

Entropy base is **nats** (matches the documented `dH/dp=log((1-p)/p)`); base only rescales the loss (absorbed
by λ) and never affects the entropy **ordering** used by surgery. **Per-site equal weighting** (mean over
sites of each site's per-neuron mean) is intentional and matches the doc's `1/|L| Σ_ℓ 1/N_ℓ Σ_i` — do **not**
replace with a single global mean over all neurons (which would over-weight wider layers) `[R28]`.

### 4.5 `training/schedule.py`
```python
class TemperatureSchedule:                               # callable: step:int → tau:float
    def __init__(self, kind, tau_start, tau_end, total_steps): ...   # validate tau_start>0, tau_end>0
def checkpoint_steps(total_steps: int, steps_per_epoch: int, every_epochs: int) -> list[int]:
    # pure: {0} ∪ {multiples of every_epochs*steps_per_epoch} ∪ {total_steps-1}, sorted, de-duplicated  [R11]
```
- `tau(0)=tau_start`; for non-constant kinds `tau(total_steps-1)=tau_end` (exponential = geometric interp,
  linear = lerp). `constant` returns `tau_start` (`tau_end` unused).
- **`total_steps<=1` `[R15]`:** `tau(step)=tau_start` for all `step`; the `(total_steps-1)` denominator is
  guarded; endpoint contract waived. `step>=total_steps` clamps to the last value.
- **`tau(step)>0` for all `step`** across all kinds given a valid config (tested) `[R16]`.

### 4.6 `training/logging.py`
```python
class Logger(Protocol):
    def log_scalars(self, values: Mapping[str, float], step: int) -> None: ...
    def log_image(self, name: str, path: Path, step: int | None = None) -> None: ...
    def log_video(self, name: str, path: Path, step: int | None = None) -> None: ...   # .gif file path
    def log_config(self, config: Mapping[str, object]) -> None: ...
    def finish(self) -> None: ...
```
- `NullLogger` (no-ops; default in library code) · `InMemoryLogger` (records to inspectable dicts; tests
  assert on it) · `WandbLogger(wandb_config, run=None)` — **lazily** `import wandb`; if `run is None` it calls
  `wandb.init(...)`, else it **attaches to the already-active `run`** (no second init) `[R3]`. `log_image`/
  `log_video` pass the saved **file path** to `wandb.Image` / `wandb.Video` (gif, no ffmpeg) `[R10]`.
- **wandb is imported only inside `WandbLogger` and the sweep glue, and only lazily** so importing the pure
  helpers never requires wandb `[R19]`. Tests never construct `WandbLogger`.

### 4.7 `training/checkpoint.py`
`save_checkpoint(model, step, config, dir) -> Path` writes `state_dict` (incl. `SelectiveReLU.mode` buffers)
+ `step` + serialized `ExperimentConfig`. `load_checkpoint(path, model) -> int` does a **strict** load into an
architecturally-identical model and returns `step`. Callers rebuild a fresh `ReLUMLP` from the checkpoint's
serialized `ModelConfig` before loading `[R26]`.

### 4.8 `training/trainer.py`
```python
@dataclass(frozen=True)
class StepMetrics: step:int; epoch:int; task_loss:float; reg_loss:float; total_loss:float; tau:float   # [R18]
@dataclass
class TrainResult: model: ReLUMLP; history: list[StepMetrics]; checkpoint_paths: list[Path]

def train(model, train_loader, val_loader, config, logger) -> TrainResult: ...
```
- **`total_steps` derivation `[R2]`:** built **after** loaders exist: `steps_per_epoch = len(train_loader)`
  (which already reflects the val carve-out and `drop_last`); `total_steps = epochs * steps_per_epoch`. The
  `TemperatureSchedule` is constructed **here** (not in config) from `total_steps`, and is **rebuilt per
  sweep run** since the sweep varies `epochs`/`batch_size`.
- Per step: `out=model(x); task=CE(out.logits,y); reg=sign_consistency_loss(out.pre_activations, tau(step),
  eps); total=task+λ·reg; total.backward(); (grad_clip?); opt.step()`. Logs `{loss/task, loss/reg,
  loss/total, tau}` each step and appends `StepMetrics` to `history` (so plots/tests don't need the logger).
- Checkpoints saved at exactly `checkpoint_steps(total_steps, steps_per_epoch, every_epochs)`.

### 4.9 `data/datasets.py`
```python
def make_dataloaders(config) -> tuple[DataLoader, DataLoader, DataLoader]:   # train, val(selection), test
def select_probe_neurons(model, num, seed) -> list[tuple[str, int]]          # fixed (site, idx) set
def make_probe_batch(loader, size, seed) -> Tensor                           # fixed inputs across checkpoints & runs
```
- MNIST/CIFAR via torchvision, **`ToTensor` + per-dataset normalize, NO train-time augmentation** `[R30]`
  (only seeded shuffling + init are stochastic). Downloaded to `data_dir` (git-ignored).
- **val carved from the train split** via `random_split` with an **explicit `torch.Generator(split_seed)`**
  `[R8]`. `test` is the canonical held-out split.
- **Determinism `[R8]`:** **train** loader shuffles with a seeded generator; **val/test** use
  `shuffle=False, drop_last=False` (so `collect_pre_activations` concatenates in a stable order). `train`
  honors `DataConfig.drop_last`.
- `synthetic`: Gaussian-input `TensorDataset` with random labels — **no network**; used by tests.
- Probe neurons/inputs are seeded and **fixed across checkpoints and across sweep runs** so a GIF reflects
  weight evolution only `[R25]`.

### 4.10 `analysis/`
```python
# statistics.py
def collect_pre_activations(model, loader, device) -> dict[str, Tensor]:
    # model.eval(); with torch.no_grad(): for x,_ in loader: out=model(x.to(device)); append out.pre_activations[i]
    # cat over batches → {site: [M, N]}.  Uses the SAME model(x) forward path as surgery eval ⇒ bit-exact  [R7]
@dataclass(frozen=True)
class NeuronStats: site:str; index:int; q:float; entropy:float; mean_pre:float
def neuron_stats(pre_by_site: dict[str, Tensor]) -> list[NeuronStats]   # q via hard_fraction_positive; entropy via sign_entropy
@dataclass(frozen=True)
class FrameStats: step:int; probe_activations: dict[tuple[str,int], Tensor]; q_by_site: dict[str, Tensor]
def collect_history(checkpoint_paths, model_factory, probe_batch, val_loader, probe_neurons, device) -> list[FrameStats]:
    # for each checkpoint: model = model_factory(); load_checkpoint(path, model); no_grad forward of probe_batch
    #   → probe-neuron pre-activations; and q_i over val_loader. One FrameStats per checkpoint.  [R11]
# selection.py
def rank_by_entropy(stats) -> list[NeuronStats]    # ascending H(q); deterministic tie-break by (entropy, site, index)
def rank_random(stats, seed) -> list[NeuronStats]  # control baseline (seeded)
def make_k_grid(total: int, num_k: int) -> list[int]:
    # sorted(set(round(i*total/(num_k-1)) for i in range(num_k))); always includes 0 and total; realized
    # length == number of UNIQUE points (may be < num_k on tiny models)  [R5]
def select_topk(ranked, k) -> list[NeuronStats]    # ranking & k are GLOBAL across all sites
def assign_modes(selection, tie_break) -> dict[str, Tensor]:
    # per site: full-length [N_site] int64 tensor initialized to RELU(0); selected neurons set to
    #   ZERO if q<0.5, IDENTITY if q>0.5, tie_break if q==0.5.  Un-selected stay RELU.  [R14][R17]
```

> **Surgery commitment vs eliminability `[R17]`.** `assign_modes`' 0.5 rule chooses *which* collapse
> (`ZERO`/`IDENTITY`) for a selected neuron — it is the commitment direction, **not** an eliminability test.
> **I2 losslessness holds only for neurons with exact `H(q)=0` (`q∈{0,1}`)**; for selected neurons with
> interior `q` the k-sweep measures the resulting accuracy loss *by design*. The method doc's tolerance `ε`
> corresponds to how far down the ascending-entropy ranking one chooses to cut.

### 4.11 `surgery/apply.py`
```python
def apply_modes(model, modes_by_site: dict[str, Tensor]) -> None:   # SelectiveReLU.set_modes per site (full [N] tensor)
def evaluate_accuracy(model, loader, device) -> float
@dataclass(frozen=True)
class KPoint: k:int; val_acc:float; test_acc:float                  # [R18]
def k_sweep(model, ranked, k_grid, val_loader, test_loader, device) -> list[KPoint]:
    # work = copy.deepcopy(model) ONCE; for k in k_grid: apply_modes(work, assign_modes(select_topk(ranked,k)));
    #   record val/test acc.  The canonical `model` is provably never mutated (deepcopy, not reset).  [R5][R21]
```

### 4.12 `viz/`
- **Headless rendering `[R10]`:** `matplotlib.use("Agg")` **before** importing `pyplot`; never `plt.show()`;
  always `savefig` + `plt.close(fig)`. `plots.py` functions return saved figure `Path`s.
- `animation.py` builds a **`.gif`** via imageio's **pillow** plugin from per-`FrameStats` frames; the gif
  path is logged via `WandbLogger.log_video` → `wandb.Video` (no ffmpeg/mp4 path) `[R10]`.
- Every output is saved under `runs/<id>/` **and** logged to wandb. Artifact list in §5.

### 4.13 `experiments/`
```python
# run.py
def run_experiment(config: ExperimentConfig, logger: Logger | None = None) -> ExperimentResult
@dataclass
class ExperimentResult:                                              # [R18]
    model: ReLUMLP; train_result: TrainResult; neuron_stats: list[NeuronStats]
    k_points: list[KPoint]; random_k_points: list[KPoint]; frames: list[FrameStats]
    artifact_paths: dict[str, Path]   # keys: loss_curves, mean_pre_dist, entropy_map, per_layer_entropy,
                                      #   acc_vs_k, soft_vs_hard, activation_gif, qi_bimodality_gif
# sweep.py  (wandb imported LAZILY inside the glue only)
def build_sweep_config(grids: Mapping[str, list], method: str) -> dict:   # [R20]
    # {"method": method,
    #  "metric": {"name": "val/acc", "goal": "maximize"},     # name MUST match a trainer/run-logged key
    #  "parameters": {k: {"values": v} for k, v in grids.items()}}     # keys: lr, epochs, batch_size, lam
def config_from_wandb(wandb_config: Mapping[str, object]) -> ExperimentConfig:   # PURE, unit-tested  [R9]
    # flat→nested key map: lr→OptimConfig.lr, epochs→TrainConfig.epochs,
    #   batch_size→DataConfig.batch_size, lam→RegConfig.lam; all other fields from a documented BASE
    #   default ExperimentConfig; seed from BASE; UNKNOWN keys raise KeyError; MISSING swept keys raise.
def sweep_entry() -> None:                                            # wandb.agent target  [R3]
    # import wandb; wandb.init(); cfg = config_from_wandb(dict(wandb.config));
    #   run_experiment(cfg, logger=WandbLogger(cfg.wandb, run=wandb.run))   # attaches; no 2nd init
def launch_sweep(grids, method, count) -> str:                        # wandb.sweep + wandb.agent (network glue; not unit-tested)
```
- `run_experiment` builds its own `WandbLogger` only when called standalone with no `logger`; in a sweep,
  `sweep_entry` injects the attached `WandbLogger`, so **exactly one `wandb.init()` per run** `[R3]`.
- CLI: `python -m kill_nonlinearities.experiments.run --config ...`.

---

## 5. Data flow & artifacts (`run_experiment`)

1. Seed everything (torch, numpy, random); resolve device (§8 recipe).
2. `make_dataloaders` → train/val/test; build `ReLUMLP`; `select_probe_neurons` + `make_probe_batch`.
3. `train(...)` → `TrainResult` (history + checkpoints). Scalars streamed to wandb live.
4. `collect_pre_activations(model, val_loader)` → `neuron_stats` → `q_i`, `H(q_i)`, mean pre-act.
5. `rank_by_entropy` (+ `rank_random` baseline); `k_grid = make_k_grid(total, num_k)`;
   `k_sweep` on **val and test**.
6. Soft `p_i` (final τ, on a fixed val batch) vs hard `q_i` agreement.
7. `collect_history(checkpoint_paths, …)` → frames for both GIFs.
8. Render & log all artifacts:

| Artifact (key) | Source | Notes |
| --- | --- | --- |
| Loss curves (`loss_curves`) | `history` | task/reg/total; also live `log_scalars` |
| Mean-pre-activation dist (`mean_pre_dist`) | `neuron_stats` | one panel per layer (scales differ) |
| Activation GIF (`activation_gif`) | `frames` (probe neurons × probe batch) | fixed neurons+inputs; histogram grid |
| `q_i` bimodality GIF (`qi_bimodality_gif`, **R4**) | `frames` (q over val) | histogram of `q_i` per frame |
| Sign-entropy map (`entropy_map`) | `neuron_stats` | per-layer sorted bar / heatmap (highest/lowest) |
| Per-layer mean entropy (`per_layer_entropy`, **R4**) | `neuron_stats` | one value per layer |
| Acc-vs-k val **and** test, entropy-order (`acc_vs_k`) | `k_points` | secondary x-axis `k/total` (**R5 absorbed here**) |
| Acc-vs-k random-order baseline (**R1**) | `random_k_points` | overlaid control |
| Lossless-on-selection-set prefix marker (**R2**) | count of neurons with `q_i` **exactly** ∈ {0.0,1.0} | annotation on the **val** curve only `[R17]` |
| Soft `p_i` vs hard `q_i` scatter (`soft_vs_hard`, **R3**) | step 6 | validates τ-annealing delivered hard consistency |
| λ trade-off curve | wandb sweep aggregation | accuracy vs λ / vs fraction-eliminable |

**Dropped: R5** (fraction linearized = `k/total`; shown as the acc-vs-k secondary axis).

---

## 6. Correctness invariants (→ hard tests)

- **I1 — k=0 ≡ trained model.** All modes `RELU` ⇒ `SelectiveReLU(z)` equals `torch.relu(z)` (`torch.equal`),
  so the surgical model at `k=0` reproduces the trained model's logits **bit-for-bit**.
- **I2 — selection-set losslessness.** Converting **only** neurons with **true** `H(q_i)=0` (i.e. `q_i`
  **exactly** ∈ {0,1} measured on the selection set with the strict `z>0` predicate) to their `ZERO`/`IDENTITY`
  mode leaves the logits **bit-for-bit unchanged on that same selection set** (`torch.equal`). Boundary
  convention `[R6]`: `q` uses strict `>`; a `q=0` (ZERO) neuron is lossless even if some `z==0` in the batch
  because `relu(0)=0`; a `q=1` (IDENTITY) neuron requires **all** `z>0` strictly (if any `z==0`, `q<1`, so it
  is not entropy-0 and never IDENTITY-selected). The test **must not** use `select_topk` — it filters to
  exactly `{H(q_i)=0}` — and is constructed (untrained, direct weights/biases) to guarantee **≥1 ZERO and ≥1
  IDENTITY** neuron so it is **non-vacuous** `[R6]`. Exact **only on the selection set** — test data may differ
  (the val-vs-test acc-vs-k curve makes this gap visible).
- **I3 — math identities** (properties + closed-form + `gradcheck`, float64): `binary_entropy` forward `=0` at
  `p∈{0,1}` (assert the **value** only; the endpoint **gradient is intentionally NaN/undefined** — see the
  saturation test) `[R12]`; `gradcheck` and `gradient = log((1-p)/p)` asserted on the **open interval**
  `p∈[0.05,0.95]` only `[R12]`; `H(½)=ln2`, symmetric; `soft_sign` range/monotone/`z=0`→½ and `τ→0` limit;
  `sign_consistency_loss` ≥ 0, batch-/neuron-permutation invariant **via `torch.testing.assert_close`** (reduction
  order is not bit-stable — **not** `torch.equal`) `[R13]`, matches a hand-computed value (with interior τ/z),
  gradients flow.

---

## 7. Test plan

Layered; all offline and CPU-deterministic. wandb forced `disabled` via an autouse `conftest.py` fixture that
also sets `MPLBACKEND=Agg` `[R10]`.

**Unit** (`tests/unit/`)
- `soft_sign`: range `(0,1)`; monotone in `z`; `z=0→0.5`; saturates for large `|z|/τ`; **raises `ValueError` for
  `τ<=0`** `[R16]`; `gradcheck` (float64, interior z).
- `binary_entropy`: forward `H(0)=H(1)=0` exact, `H(0.5)=ln2`, `H(p)=H(1-p)`, closed-form at `p=0.25`; on a tensor
  **mixing endpoints and interior** values → finite, exact at ends `[R22]`; `gradcheck` + `grad=log((1-p)/p)` on
  `p∈[0.05,0.95]` only `[R12]`.
- **Loss saturation (BLOCKER guard `[R1]`):** `sign_consistency_loss` with saturating pre-activations
  (`|z|/τ ≈ 20`, float32) → **finite** loss and **finite, non-NaN/Inf gradients on every pre-activation**.
- `batch_fraction_positive`/`hard_fraction_positive`: shapes `[N]`; values vs manual; `q∈[0,1]`; strict `>` at
  `z==0` (→ counts as negative).
- `sign_consistency_loss`: scalar ≥ 0; equals hand-computed value on a 2-site toy input with **interior** τ≈1,
  `|z|/τ≲5` `[R-nit]`; single-site == that site's mean entropy; gradient flows to all pre-activations;
  batch- **and** neuron-permutation invariance via `assert_close` (two separate cases) `[R13]`; the per-neuron
  entropy **vector** is invariant under the inverse neuron permutation (`assert_close` — the
  reduction order over a column-permuted tensor is not bit-stable) `[R13]`.
- `SelectiveReLU`: all-`RELU` ≡ `torch.relu` (`torch.equal`) **I1**; `ZERO`→zeros; `IDENTITY`→`z` exact; mixed
  per-neuron; broadcast `[N]` over `[B,N]`; `set_modes` validation (shape/dtype/range) raises; **gradient routing
  by exact analytic assertion** (`z.grad==0` ZERO, `==1` IDENTITY, `==(z>0).float()` RELU) `[R12]`; **state_dict
  round-trip with a non-default mode** and **`.to(torch.float64)` keeps mode comparisons working** `[R22][R27]`.
- `TemperatureSchedule`: endpoints (`tau(0)=tau_start`, `tau(total_steps-1)=tau_end` non-constant); monotone; each
  `kind`; **`total_steps<=1`** → `tau_start`, no div-by-zero `[R15]`; **`tau(step)>0` ∀ step** `[R16]`; validates
  `tau_start,tau_end>0`.
- `checkpoint_steps`: pure; includes 0 and `total_steps-1`; de-duplicated; matches a hand-enumerated set `[R11]`.
- `make_k_grid`: includes 0 and `total`; `== sorted(set(...))`; on `total=4, num_k=21` returns the 5 unique
  points (assert membership/uniqueness, **not** `len==num_k`) `[R5]`.
- `selection`: ascending-entropy order; **real tie** (≥2 neurons with byte-identical entropy across sites) ordered
  by `(entropy, site, index)` `[R29]`; `select_topk` boundaries `k=0`/`k=total`; `assign_modes` returns full-width
  `[N]` tensors (`q<0.5`→ZERO, `q>0.5`→IDENTITY, tie→config; unselected RELU) `[R14]`; `rank_random` deterministic.
- `config_from_wandb`: flat dict → expected nested `ExperimentConfig` (full equality); unknown key raises;
  missing swept key raises `[R9]`. `build_sweep_config`: grids → exact expected dict incl. `metric` `[R20]`.
- `import kill_nonlinearities.experiments.sweep` does **not** import wandb `[R19]`.

**Functional** (`tests/functional/`)
- `ReLUMLP.forward`: logits `[B,C]`; `len(pre_activations)==#hidden`; shapes; equals a manual layer-by-layer
  recompute (proving captured pre-acts are the true activation inputs) `[R7]`.
- Surgical model at **k=0** reproduces base logits exactly (**I1**) inside a real `ReLUMLP`.
- **k=total** through the full path: every neuron converted → finite logits, runs clean `[R22]`.
- `collect_pre_activations`/`neuron_stats`: stats vs hand-computed on a tiny loader; uses `model(x).pre_activations`.
- One-batch overfit: a few steps reduce task loss; **annealed-τ steps keep all grads/params finite** `[R1]`.
- Logger protocol: `NullLogger` no-ops; `InMemoryLogger` records expected keys/steps.
- Checkpoint round-trip: save→load reproduces logits via `torch.equal`, returns the saved `step`, and the
  deserialized config equals the original `[R31]`; round-trips a **non-default** mode buffer.
- `k_sweep` non-mutation: snapshot canonical modes+logits, run sweep, assert modes still all-`RELU` and logits
  `torch.equal` to the snapshot `[R21]`.
- Soft-vs-hard agreement (**R3**) on a fixed batch vs hand-computed `[R22]`.
- Probe fixity: `select_probe_neurons`/`make_probe_batch` deterministic under seed (`torch.equal`) `[R22]`.
- `N=1` and **`N=0` (empty) site** through loss/stats/ranking behave sanely (documented) `[R22]`.

**Integration** (`tests/integration/`)
- End-to-end `run_experiment` on **synthetic** data (tiny, seeded, wandb `disabled`, λ>0): runs clean;
  checkpoints exactly at `checkpoint_steps`; finite losses; `k_sweep` returns val+test acc for every grid point;
  all `artifact_paths` files exist and are non-empty; **both GIFs have `frame_count == len(checkpoint_steps)`**
  `[R11]`; at least one viz test **extracts the matplotlib Axes data and asserts it equals `k_points`
  accuracies** `[R23]`.
- **I2** (own test, untrained model with direct large +/− biases guaranteeing ≥1 ZERO and ≥1 IDENTITY entropy-0
  neuron; filter to `{H(q)=0}`, **not** `select_topk`): converting them is `torch.equal`-lossless on the selection
  set `[R6]`.
- **Determinism `[R8]`:** two same-config runs in one process (CPU, `num_workers=0`,
  `torch.use_deterministic_algorithms(True)`, `set_num_threads(1)`) → identical `history` losses and `q_i`
  (`torch.equal`).
- `config_from_wandb` end-to-end: a wandb-style dict → expected `ExperimentConfig` (no network).

**Explicitly untested / smoke-only `[R23]`** (`# pragma: no cover` with rationale): `WandbLogger` body and
`wandb.Image`/`wandb.Video` calls; `launch_sweep` and the network portion of `sweep_entry` (its pure
config-translation is factored into `config_from_wandb` and tested); genuinely display-only plotting branches.
Viz is otherwise verified by file-output + the one data-correctness Axes assertion above.

**Coverage:** ≥ 90% gate; ~100% target on `regularization/`, `models/`, `analysis/`, `surgery/`.

---

## 8. Cross-cutting concerns & interactions

- **Determinism recipe `[R8]`:** seed `torch`/`numpy`/`random`; `torch.use_deterministic_algorithms(True)`;
  `torch.set_num_threads(1)` and `num_workers=0` in the determinism test; explicit `torch.Generator(split_seed)`
  for `random_split`; train shuffles with a seeded generator; val/test `shuffle=False`; `worker_init_fn` when
  `num_workers>0`. Exactness is asserted on CPU, one process, identical config only.
- **Numerical stability:** loss clamps `p∈[eps,1-eps]` `[R1]`; `xlogy` keeps analysis entropy exact at `q∈{0,1}`;
  `gradcheck` interior + float64; `soft_sign` guards `τ>0`; schedule guards `τ>0` and `total_steps<=1`.
- **`SelectiveReLU` buffer:** int64; `set_modes`-validated; survives `state_dict`/`.to`; comparisons via
  `int(ActivationMode.X)` `[R27]`. k-sweep operates on a `deepcopy`, never mutating the canonical model `[R5]`.
- **Bit-exactness chain (I1/I2):** analysis and surgery-eval both go through the *same* `model(x)` forward and
  the *same* `ForwardOutput.pre_activations` tensors `[R7]`.
- **Splits:** train / val(selection) / test; entropy & selection on val; test only for the acc-vs-k generalization
  curve and the I2 caveat.
- **wandb & CI:** core imports wandb only inside `WandbLogger`/sweep glue, lazily `[R19]`; exactly one
  `wandb.init()` per run `[R3]`; tests use `InMemoryLogger`; an autouse fixture sets `WANDB_MODE=disabled` +
  `MPLBACKEND=Agg` `[R10]`.
- **Headless render `[R10]`:** `matplotlib.use("Agg")` before `pyplot`; gif via imageio+pillow; `wandb.Video`
  takes the gif path.
- **Sweep ↔ schedule ↔ probe:** the τ schedule is rebuilt per run from the realized `total_steps` `[R2]`; the
  probe neurons/batch and `eval_batch_size` are invariant to the `batch_size` sweep `[R25]`.
- **Memory:** `collect_pre_activations` holds `[M,N]` per site (M ≤ val size ≤ 10k, N ≤ 512) — fine on CPU.

## 9. Risks & mitigations

| Risk | Mitigation |
| --- | --- |
| Entropy backward NaN at saturation (**verified**) | loss clamps `p∈[eps,1-eps]`; dedicated finite-gradient test `[R1]` |
| τ-schedule endpoint/total_steps drift | `total_steps=epochs*len(train_loader)`, rebuilt per run; endpoint test `[R2]` |
| wandb double-init / orphaned sweep config | single `wandb.init` in `sweep_entry`; `WandbLogger` attaches `[R3]` |
| λ too large collapses the net | λ config; sweep incl. λ=0 baseline; trade-off curve shows collapse |
| τ leaves a soft/hard gap | R3 soft-vs-hard scatter quantifies it; anneal `tau_end` low |
| I2 vacuous / boundary wrong | untrained construction forcing ≥1 ZERO+≥1 IDENTITY; strict `>`; no `select_topk` `[R6]` |
| Headless render / missing gif deps | Agg backend; pillow+imageio runtime deps; gif (not mp4) `[R10]` |
| Sweep combinatorics explode | explicit grids; documented run-count; modest defaults |

## 10. Documentation updates (part of the work)

- `docs/architecture/overview.md`: realized layout (`data/`, `viz/`, `experiments/`, `surgery/`),
  `SelectiveReLU`, `ForwardOutput(eq=False)`.
- `docs/research/README.md`: flip roadmap 1a + surgery to in-progress; describe masked-activation surgery (vs
  folding); add the analyses; record the **selection-set-only** losslessness caveat + the loss clamp; add
  experiment-log entries after real runs.
- `docs/development/`: `setup.md` (wandb login + new deps, `MPLBACKEND=Agg`, CUDA note), `tooling.md` (wandb),
  `testing.md` (unit/functional/integration layout, determinism recipe, offline-wandb + Agg fixture, the
  pragma-excluded surfaces).
- `AGENTS.md` + root `README.md`: status → phase 1a in progress; new deps.
- Add a CI workflow (`uv sync` + `just check` with `WANDB_MODE=disabled`, `MPLBACKEND=Agg`) `[R10]`.
- Link this spec from `docs/README.md`.

## 11. Implementation milestones (ordered; detailed in the plan)

1. `config` + `regularization/` (loss clamp; gradcheck/saturation tests).
2. `models/` (`SelectiveReLU`, `ReLUMLP`, `ForwardOutput`) (+ **I1**, gradient-routing tests).
3. `training/` (`schedule`+`checkpoint_steps`, `logging`, `checkpoint`, `trainer`; total_steps derivation).
4. `data/` (synthetic + torchvision; probe; deterministic splits/loaders).
5. `analysis/` (`statistics`, `collect_history`, `selection`, `make_k_grid`) + `surgery/` (+ **I2**, non-mutation).
6. `viz/` (Agg plots + gif) (+ smoke + one Axes-data assertion).
7. `experiments/run.py` + integration tests (synthetic, end-to-end, determinism).
8. `experiments/sweep.py` (lazy wandb; `config_from_wandb`/`build_sweep_config` pure tests).
9. Docs updates + CI workflow; real MNIST + CIFAR runs (λ>0); experiment-log entries.

## 12. Open questions

None outstanding. Decisions settled: datasets (MNIST→CIFAR, agnostic core), wandb logging + **wandb Sweeps**,
single-run-core + sweep, internal invariants. Analyses: R1–R4 in, R5 dropped, R2 reframed to selection-set-only
with a val-vs-test acc-vs-k curve. All four review blockers and the major/minor/nit findings are folded into
the sections above (tagged `[Rxx]`).
