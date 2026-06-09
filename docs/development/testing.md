# Testing

[← Development](README.md) · [← Documentation hub](../README.md)

We use [`pytest`](https://docs.pytest.org/) for tests, [`hypothesis`](https://hypothesis.readthedocs.io/)
for property-based testing, and [`pytest-cov`](https://pytest-cov.readthedocs.io/) for
coverage. Configuration is in [`pyproject.toml`](../../pyproject.toml); shared fixtures and
Hypothesis profiles are in [`tests/conftest.py`](../../tests/conftest.py).

```bash
just test                 # run everything (with coverage)
just test -k version      # filter by name
just test -m "not slow"   # skip slow-marked tests
just cov                  # also write an HTML report to htmlcov/
```

## Conventions

- **Tests live in `tests/`**, split into three layers (all offline, all CPU-deterministic):
  - **`tests/unit/`** — pure functions and single classes (the math, schedule, selection,
    config translation, `SelectiveReLU`). Fast; property tests welcome here.
  - **`tests/functional/`** — small real `ReLUMLP`s end-to-end pieces (forward recompute,
    collect/stats, checkpoint round-trip, k-sweep non-mutation).
  - **`tests/integration/`** — `run_experiment` on tiny seeded **synthetic** data, the **I2**
    losslessness test, and the determinism test.
- **Import mode is `importlib`** and the package is installed by `uv`, so tests import it as
  `import kill_nonlinearities` — no `sys.path` hacks.
- **Coverage gate: ≥ 90%** (`fail_under = 90` in `[tool.coverage.report]`), with branch
  coverage on. The run fails if coverage drops below the threshold.
- **Markers are strict** (`--strict-markers`): register any new marker in
  `[tool.pytest.ini_options].markers` before using it. `slow` is registered as an example.

## Property-based testing with Hypothesis

The math at the heart of this project (soft sign, batch fraction, entropy, the loss) is made
of **pure functions** — ideal for property tests. Prefer asserting *properties* over
hand-picked examples. Useful properties to keep in mind when those functions land:

- `binary_entropy(p)` is maximal at `p = 0.5`, zero at `p ∈ {0, 1}`, and symmetric:
  `H(p) == H(1 - p)`.
- `soft_sign(z, tau)` lies in `(0, 1)`, is monotonic in `z`, and `== 0.5` at `z == 0`.
- the loss is non-negative and invariant to neuron permutation.

See [the method](../research/README.md#the-method) for the definitions these properties come
from.

### Hypothesis profiles

[`tests/conftest.py`](../../tests/conftest.py) registers two profiles and selects one via the
`HYPOTHESIS_PROFILE` environment variable:

| Profile | `max_examples` | Use |
| --- | --- | --- |
| `dev` (default) | 25 | fast local runs |
| `ci` | 500 (no deadline) | thorough; set `HYPOTHESIS_PROFILE=ci` in CI |

```bash
HYPOTHESIS_PROFILE=ci just test    # thorough search
```

## Offline by default: the wandb + matplotlib setup

[`tests/conftest.py`](../../tests/conftest.py) makes the whole suite hermetic: at import time
it sets `WANDB_MODE=disabled` (no network, no login) and `MPLBACKEND=Agg` (no display) for
every test via `os.environ.setdefault`. Tests therefore never construct a `WandbLogger` (they
use `InMemoryLogger`) and never open a plotting window. Viz tests assert on **saved file
paths** (the figure/GIF exists and is non-empty) plus one data-correctness check that extracts
the matplotlib `Axes` data and asserts it equals the `k_points` accuracies.

## Determinism recipe

Bit-exact reproducibility is asserted **on CPU, in one process, with identical config only**
(it does *not* hold across sweep points that vary `batch_size`/`epochs`). The determinism test
applies:

- seed `torch`, `numpy`, and `random` (`run_experiment` re-seeds all three first thing);
- `torch.use_deterministic_algorithms(True)` and `torch.set_num_threads(1)`;
- `num_workers=0`, an explicit `torch.Generator(split_seed)` for the val carve-out
  (`random_split`), a seeded generator for the **train** shuffle, and `shuffle=False` for
  val/test (so `collect_pre_activations` concatenates in a stable order);
- two same-config runs then produce identical `history` losses and `q_i` (`torch.equal`).

Numerical comparisons elsewhere use `torch.testing.assert_close` (tolerances), **except**
where bit-exactness is the property under test (I1/I2, all-`RELU` ≡ `torch.relu`), which use
`torch.equal`. The loss permutation-invariance tests (including the per-neuron entropy
**vector** under the inverse permutation) use `assert_close` because the reduction order over
a column-permuted tensor is not bit-stable.

## Explicitly untested surfaces (`# pragma: no cover`)

A few surfaces are network/display-only and are marked `# pragma: no cover` with a rationale,
so the ≥ 90% coverage gate (with ~100% targeted on `regularization/`, `models/`, `analysis/`,
`surgery/`) stays honest:

- the **`WandbLogger`** body and the `wandb.Image` / `wandb.Video` calls;
- **`launch_sweep`** and the network portion of `sweep_entry` (its pure config translation is
  factored into `config_from_wandb`, which *is* unit-tested);
- the `run.py` CLI glue (`main` / `__main__`) and the lazy `WandbLogger` construction in
  `run_experiment` when no logger is supplied.

## The scaffold tests

[`tests/test_scaffold.py`](../../tests/test_scaffold.py) originally proved only that the
harness works (import + a trivial property test). As the real suite under `tests/unit/`,
`tests/functional/`, and `tests/integration/` lands, the scaffold test is superseded — it has
no connection to the research idea.

## Numerical testing tips (for when real code lands)

- Use `torch.testing.assert_close` for tensor comparisons (tolerances, not `==`).
- Seed with `torch.manual_seed` in fixtures for determinism; let Hypothesis explore *inputs*,
  not RNG state.
- Test gradients flow (e.g. `loss.backward()` produces finite, non-zero grads on the
  pre-activations) — the whole point is that the regularizer is differentiable.

---

[← Development](README.md) · [Contributing →](contributing.md)
