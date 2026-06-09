# Setup

[← Development](README.md) · [← Documentation hub](../README.md)

## Prerequisites

| Tool | Why | Install |
| --- | --- | --- |
| [`uv`](https://docs.astral.sh/uv/) | packaging, venv, Python install | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| [`just`](https://just.systems) | task runner | see [just install docs](https://just.systems/man/en/packages.html) (or `uv tool install rust-just`) |

`ruff`, `ty`, `pytest`, `hypothesis`, `pytest-cov`, and `pre-commit` are **dev
dependencies** — `uv` installs them; you do not install them globally.

### Runtime dependencies

Phase 1a adds runtime dependencies (in `[project].dependencies`, not dev-only, because the
integration tests import them end-to-end): **`torchvision`** (MNIST/CIFAR), **`wandb`**
(experiment tracking + sweeps), **`matplotlib`** (figures), and **`imageio`** + **`pillow`**
(GIFs). `numpy` arrives transitively. `uv sync` installs them all.

## Python version

The project pins **Python 3.14** via [`.python-version`](../../.python-version), with
`requires-python = ">=3.13"` as the supported floor.

- `torch` 2.12 ships CPython 3.14 wheels (the version this project pins); check the
  [PyTorch release notes](https://github.com/pytorch/pytorch/releases) for the current
  support matrix.
- `uv` will install the interpreter automatically on first sync. At the time of writing,
  `uv`'s build index serves **3.14.0rc3** (3.14 final isn't in the index yet); the pin
  tracks 3.14.x as `uv` updates. If you hit a wheel that lacks a 3.14 build, drop the pin to
  `3.13` (`uv python pin 3.13`) — everything still works on the 3.13 floor.

## Create the environment

```bash
just setup        # == uv sync  +  pre-commit install
# or step by step:
uv sync           # create .venv, install deps + dev tools, install the project (editable)
uv run pre-commit install
```

Verify:

```bash
just check        # lint + type-check + test, all green
```

## Weights & Biases (wandb)

Experiment tracking and hyperparameter sweeps use [wandb](https://wandb.ai/). For **real
runs** (milestone 9), log in once:

```bash
uv run wandb login        # paste your API key from https://wandb.ai/authorize
```

You do **not** need an account to develop: the test suite forces `WANDB_MODE=disabled` (no
network, no login) via the shared [`tests/conftest.py`](../../tests/conftest.py), and you can
run experiments offline with `WandbConfig.mode="offline"` (logs to a local `wandb/` dir) or
skip logging entirely with `mode="disabled"`. Only the documented MNIST/CIFAR runs and live
sweeps need `mode="online"`.

## Headless plotting (`MPLBACKEND=Agg`)

All figures and GIFs are rendered headlessly: the code selects matplotlib's **Agg** backend
before importing `pyplot` and never calls `plt.show()`. Tests also export `MPLBACKEND=Agg`
(via `tests/conftest.py`) so they never try to open a display. If you render manually on a
headless box, export it yourself:

```bash
export MPLBACKEND=Agg
```

## Run an experiment

The runner is `kill_nonlinearities.experiments.run`; pass a nested JSON `ExperimentConfig`
(see [`configs/mnist.json`](../../configs/mnist.json)). The standalone CLI uses a no-op
logger, so it needs no wandb account and downloads MNIST to `./data` on first run:

```bash
uv run python -m kill_nonlinearities.experiments.run --config configs/mnist.json
```

Artifacts (loss curves, entropy map, acc-vs-k, both GIFs, …) land under `runs/<name>/`.
`./data/`, `runs/`, and `wandb/` are all git-ignored.

## CPU vs. CUDA `torch`

By default the project installs the **CPU** build of `torch`. This is deliberate: the plain
PyPI `torch` wheel on Linux is the **CUDA build**, which pulls in multi-gigabyte NVIDIA
packages. The CPU build keeps setup light and portable, and is plenty for the phase-1 toy
experiments. This is configured in [`pyproject.toml`](../../pyproject.toml):

```toml
[tool.uv.sources]
torch = { index = "pytorch-cpu" }

[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true
```

**To use a GPU (CUDA) build**, change only the index **URL** and re-sync — e.g. for CUDA
13.0. Keep the index `name = "pytorch-cpu"`, because `[tool.uv.sources]` in
[`pyproject.toml`](../../pyproject.toml) binds `torch` to that index *by name*; if you rename
the index you must rename it there too, or `uv sync` will fail.

```toml
[[tool.uv.index]]
name = "pytorch-cpu"   # keep this name (or rename here AND in [tool.uv.sources])
url = "https://download.pytorch.org/whl/cu130"
explicit = true
```

```bash
uv sync
```

Alternatively, delete the `[tool.uv.sources]` + `[[tool.uv.index]]` block entirely and run
`uv sync` — plain PyPI `torch` on Linux already *is* the CUDA build. Pick whichever matches
your hardware; see the [PyTorch install selector](https://pytorch.org/get-started/locally/).

## Common issues

- **`just` not found** — install it (table above) or run recipes directly, e.g.
  `uv run pytest`.
- **First `uv sync` is slow** — it downloads the interpreter and `torch`. Subsequent syncs
  are cached.
- **`ty` reports something new after an upgrade** — `ty` is beta (0.0.x) and its diagnostics
  can change between releases; see [tooling.md](tooling.md).

---

[← Development](README.md) · [Tooling →](tooling.md)
