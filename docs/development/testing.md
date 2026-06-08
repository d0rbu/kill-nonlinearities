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

- **Tests live in `tests/`**, mirroring `src/kill_nonlinearities/`.
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

## The scaffold tests

[`tests/test_scaffold.py`](../../tests/test_scaffold.py) only proves the harness works
(import + a trivial property test). **Replace these** with real tests as the library grows —
they have no connection to the research idea.

## Numerical testing tips (for when real code lands)

- Use `torch.testing.assert_close` for tensor comparisons (tolerances, not `==`).
- Seed with `torch.manual_seed` in fixtures for determinism; let Hypothesis explore *inputs*,
  not RNG state.
- Test gradients flow (e.g. `loss.backward()` produces finite, non-zero grads on the
  pre-activations) — the whole point is that the regularizer is differentiable.

---

[← Development](README.md) · [Contributing →](contributing.md)
