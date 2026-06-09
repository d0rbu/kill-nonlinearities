# Tooling

[← Development](README.md) · [← Documentation hub](../README.md)

Every tool is configured in [`pyproject.toml`](../../pyproject.toml) (except pre-commit,
which is [`.pre-commit-config.yaml`](../../.pre-commit-config.yaml)) and driven through the
[`justfile`](../../justfile).

## `just` recipes

Run `just` (no args) to list them. Source of truth is the [`justfile`](../../justfile).

| Recipe | Does |
| --- | --- |
| `just setup` | `install` + `hooks` — full first-time setup |
| `just install` | `uv sync` — create/update `.venv` from the lockfile |
| `just hooks` | install the git pre-commit hooks |
| `just fmt` | `ruff format` then `ruff check --fix` — auto-fix in place |
| `just lint` | `ruff check` + `ruff format --check` — no writes (CI-style) |
| `just typecheck` | `ty check` |
| `just test [args]` | `uv run pytest` (extra args pass through) |
| `just cov` | tests + HTML coverage report in `htmlcov/` |
| `just check` | `lint` + `typecheck` + `test` — **the local gate** |
| `just hooks-all` | run every pre-commit hook on all files |
| `just update` | `uv lock --upgrade` |
| `just clean` | remove caches and build/coverage artifacts |

## `uv` — packaging & environments

[`uv`](https://docs.astral.sh/uv/) manages the interpreter, the virtual environment, and
dependencies. The lockfile [`uv.lock`](../../uv.lock) is committed for reproducibility.

```bash
uv add <pkg>            # add a runtime dependency
uv add --dev <pkg>      # add a dev dependency (goes in [dependency-groups].dev)
uv run <cmd>            # run a command in the project environment
uv sync                 # make .venv match the lockfile
uv lock --upgrade       # bump locked versions
```

- **Runtime deps** live in `[project].dependencies`; **dev deps** in PEP 735
  `[dependency-groups].dev`.
- The build backend is `uv_build`. The package uses a `src/` layout
  (`src/kill_nonlinearities/`) and ships a `py.typed` marker so downstream type checkers see
  its types.

## `ruff` — lint + format

[`ruff`](https://docs.astral.sh/ruff/) is both linter and formatter.

```bash
uv run ruff check          # lint
uv run ruff check --fix    # lint + autofix
uv run ruff format         # format
uv run ruff format --check # verify formatting without writing
```

Config highlights ([`[tool.ruff]`](../../pyproject.toml)):

- **`target-version = "py313"`** — matches the supported floor so `pyupgrade` (`UP`) never
  rewrites to syntax that breaks on 3.13.
- **Lint rule families:** `E`/`W` (pycodestyle), `F` (pyflakes), `I` (isort), `UP`
  (pyupgrade), `B` (bugbear), `SIM`, `C4`, `N` (naming), `PIE`, `RET`, `PTH` (use pathlib),
  `TID` (relative imports banned), `RUF`. `E501` is ignored — the formatter owns line length.
- **Formatter:** double quotes, 88 cols, formats code in docstrings.

## `ty` — type checking

[`ty`](https://docs.astral.sh/ty/) is Astral's type checker.

```bash
uv run ty check
```

Config ([`[tool.ty]`](../../pyproject.toml)): checks `src` and `tests` against Python
**3.13** (the floor), with `error-on-warning = true` so warnings fail too, and a few
diagnostics escalated to errors (`possibly-unresolved-reference`, `division-by-zero`,
`deprecated`).

> ⚠️ **`ty` is beta (0.0.x)**; rule names and diagnostics can change between releases. It is
> pinned in the dev group — when you bump it, re-run `just typecheck` and
> re-check the rule names in `[tool.ty.rules]` against the
> [current rules reference](https://docs.astral.sh/ty/reference/rules/).

## `wandb` — experiment tracking & sweeps

[Weights & Biases](https://wandb.ai/) tracks training metrics, logs artifacts (figures,
GIFs), and runs hyperparameter **sweeps** over `{lr, epochs, batch_size, λ}`.

```bash
uv run wandb login        # one-time auth for online runs
uv run wandb offline      # toggle the CLI default to offline (local logging)
WANDB_MODE=disabled ...   # no logging at all (what the test suite uses)
```

- **Lazy import.** `wandb` is imported **only inside** `WandbLogger` and the sweep glue, and
  only at call time, so importing the pure helpers (loss, analysis, config translation) never
  requires wandb. Tests never construct `WandbLogger`; they use `InMemoryLogger`.
- **Modes** (`WandbConfig.mode`): `online` (default; needs login), `offline` (local `wandb/`
  dir), `disabled` (no-op). The test suite forces `WANDB_MODE=disabled` via
  [`tests/conftest.py`](../../tests/conftest.py).
- **Artifacts** are saved under `runs/<name>/` *and* logged via `wandb.Image` (figures) and
  `wandb.Video` (GIFs — no ffmpeg, the gif path is passed directly).

## `pre-commit`

[`.pre-commit-config.yaml`](../../.pre-commit-config.yaml) runs on every commit (after
`just hooks`):

- **hygiene** (`pre-commit-hooks`): trailing whitespace, end-of-file, YAML/TOML validity,
  merge-conflict markers, large-file guard, line endings.
- **`ruff-check --fix`** then **`ruff-format`** (lint must run before format).
- **`ty`** as a `local` hook (via `uv run ty check`) — `ty` has no official pre-commit hook
  yet ([astral-sh/ty#269](https://github.com/astral-sh/ty/issues/269)).

The hooks enforce a *subset* of the gate (formatting, lint, types). The **test suite** is not
a hook — run it via `just check` / `just test` (and in CI).

```bash
just hooks-all                       # run all hooks now
uv run pre-commit autoupdate         # refresh pinned hook revisions
```

---

[← Development](README.md) · [Testing →](testing.md)
