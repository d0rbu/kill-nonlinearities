# kill-nonlinearities

> Regularize a network toward **sign-consistent pre-activations** so that nonlinearities
> which never actually "switch" can be identified — and ultimately removed.

A ReLU only does something nonlinear for a neuron when that neuron is positive for some
inputs and negative for others. If, across the data, a neuron's pre-activation keeps the
**same sign**, its ReLU is effectively linear (always-on → identity) or dead (always-off →
removable). This project adds a regularization term that *pushes* neurons toward that
sign-consistent regime, then measures how many nonlinearities become unnecessary.

The regularizer is a side-computation only — **the forward pass stays a plain ReLU**. For
each pre-ReLU activation `z` we compute a soft "fired positive" probability `σ(z/τ)`,
average it over the batch to get `pᵢ ∈ (0, 1)` per neuron, and **minimize the binary
entropy `H(pᵢ)`** to drive each neuron toward consistently-positive or consistently-negative.

➡️ **Full motivation, math, and the elimination plan:** [`docs/research/README.md`](docs/research/README.md)

## Status

🚧 **Scaffold only.** The tooling, tests harness, and documentation are set up; no research
code has been implemented yet. See the [roadmap](docs/research/README.md#roadmap).

## Quickstart

```bash
# Prerequisites: uv (https://docs.astral.sh/uv/) and just (https://just.systems)
just setup     # create the environment + install git hooks
just check     # lint + type-check + test (the full local gate)
```

Common tasks (run `just` to list them all):

| Command | What it does |
| --- | --- |
| `just install` | Sync the environment from the lockfile |
| `just fmt` | Auto-format and apply safe lint fixes |
| `just lint` | Lint + check formatting (no writes) |
| `just typecheck` | Static type check with `ty` |
| `just test` | Run the test suite (with coverage) |
| `just check` | `lint` + `typecheck` + `test` |

## Repository layout

```
kill-nonlinearities/
├── AGENTS.md                # how to work in this repo (CLAUDE.md → symlink)
├── README.md                # you are here
├── justfile                 # task runner
├── pyproject.toml           # project + uv/ruff/ty/pytest/coverage config
├── docs/                    # hierarchical documentation (start at docs/README.md)
│   ├── research/            # the method, the math, and the experiment log
│   ├── development/         # setup, tooling, testing, contributing
│   └── architecture/        # planned code structure
├── src/kill_nonlinearities/ # the package (currently a scaffold)
└── tests/                   # the test suite
```

## Documentation

- 📚 [`docs/README.md`](docs/README.md) — documentation hub
- 🔬 [`docs/research/README.md`](docs/research/README.md) — the research idea, method, and log
- 🛠️ [`docs/development/`](docs/development/README.md) — environment, tooling, testing, contributing
- 🏗️ [`docs/architecture/`](docs/architecture/README.md) — planned code structure
- 🤖 [`AGENTS.md`](AGENTS.md) — conventions for humans and coding agents

## Tooling

[`uv`](https://docs.astral.sh/uv/) (packaging) · [`ruff`](https://docs.astral.sh/ruff/)
(lint + format) · [`ty`](https://docs.astral.sh/ty/) (types) ·
[`pytest`](https://docs.pytest.org/) + [`hypothesis`](https://hypothesis.readthedocs.io/)
+ [`pytest-cov`](https://pytest-cov.readthedocs.io/) (tests) ·
[`pre-commit`](https://pre-commit.com/) (hooks) · [`PyTorch`](https://pytorch.org/).
