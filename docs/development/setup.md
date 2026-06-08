# Setup

[← Development](README.md) · [← Documentation hub](../README.md)

## Prerequisites

| Tool | Why | Install |
| --- | --- | --- |
| [`uv`](https://docs.astral.sh/uv/) | packaging, venv, Python install | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| [`just`](https://just.systems) | task runner | see [just install docs](https://just.systems/man/en/packages.html) (or `uv tool install rust-just`) |

`ruff`, `ty`, `pytest`, `hypothesis`, `pytest-cov`, and `pre-commit` are **dev
dependencies** — `uv` installs them; you do not install them globally.

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
