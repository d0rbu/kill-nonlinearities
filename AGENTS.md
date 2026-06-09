# AGENTS.md

Guidance for coding agents (and humans) working in **kill-nonlinearities**.
`CLAUDE.md` is a symlink to this file — there is one source of truth.

> **User instructions and this file take precedence over default agent behavior.** When in
> doubt, prefer what's written here and in the linked docs.

## What this project is

A research repo testing one idea: **add a regularizer that encourages a network's ReLUs to
have sign-consistent pre-activations**, so that nonlinearities which never actually switch
sign can be identified — and eventually removed. For each pre-ReLU activation `z` we compute
`σ(z/τ)` (a soft "fired positive" probability), average it over the batch to get `pᵢ`, and
**minimize the binary entropy `H(pᵢ)`** so each neuron commits to being always-positive
(ReLU ≈ identity) or always-negative (dead). The **forward pass stays a plain ReLU**; the
regularizer is a side-computation only.

📖 **Read [`docs/research/README.md`](docs/research/README.md) for the full motivation, math,
and elimination plan before touching research code.**

## Status & scope

🚧 **Phase 1a in progress.** The regularizer, models (`SelectiveReLU` + `ReLUMLP`), training,
data, analysis, **masked-activation surgery**, viz, and the experiment runner/sweep glue are
implemented per [`docs/specs/2026-06-08-phase1a-regularizer-and-surgery-design.md`](docs/specs/2026-06-08-phase1a-regularizer-and-surgery-design.md).

- The work stays **analysis-first plus masked surgery**: flip a neuron's `SelectiveReLU`
  **mode** (`ZERO`/`IDENTITY`), don't fold or structurally prune — that's a later phase. Do
  not jump ahead to linear-folding.
- Don't add features beyond what a task asks for. The [roadmap](docs/research/README.md#roadmap)
  and the [phase-1a spec](docs/specs/2026-06-08-phase1a-regularizer-and-surgery-design.md) are
  the plan of record.

## Repository map

```
AGENTS.md / CLAUDE.md     this file (CLAUDE.md → AGENTS.md symlink)
README.md                 project front door
justfile                  task runner — `just` lists recipes
pyproject.toml            project metadata + uv/ruff/ty/pytest/coverage config
.pre-commit-config.yaml   pre-commit hooks
uv.lock                   committed lockfile
docs/                     hierarchical docs — start at docs/README.md
  research/README.md         the method, the math, and the experiment log
  development/               setup · tooling · testing · contributing
  architecture/overview.md   realized code layout + key design decisions
  specs/                     dated design specs (phase-1a is the design of record)
src/kill_nonlinearities/  the package (models, regularization, training, data, analysis, surgery, viz, experiments)
tests/                    the test suite (unit / functional / integration)
```

## Environment & commands

- **`uv` for everything.** Never `pip install` into the env. Add deps with `uv add` /
  `uv add --dev`; run things with `uv run`.
- **Python 3.14** (pinned in `.python-version`; floor `>=3.13`). `uv` installs the
  interpreter on first `uv sync`.
- **`torch` defaults to the CPU build** (the PyPI Linux wheel is the multi-GB CUDA build).
  See [setup.md](docs/development/setup.md) to switch to CUDA.
- **Runtime deps** now include `torchvision`, `wandb`, `matplotlib`, `imageio`, `pillow`
  (in `[project].dependencies` — the integration tests import them end-to-end). Tests run
  offline: `tests/conftest.py` forces `WANDB_MODE=disabled` + `MPLBACKEND=Agg`.

```bash
just setup       # uv sync + install git hooks (first time)
just check       # lint + type-check + test — THE gate; run before committing
just fmt         # auto-format + safe lint fixes
just test -k x   # run a subset
```

Full recipe list: [`docs/development/tooling.md`](docs/development/tooling.md).

## Conventions (the rules)

1. **`just check` must be green before you commit.** That's `ruff` (lint + format), `ty`,
   and `pytest`.
2. **Type everything; `ty` must pass.** `ty` is beta (0.0.x) — re-verify rule names when you
   bump it ([tooling.md](docs/development/tooling.md)).
3. **`ruff` owns style.** Don't hand-format; run `just fmt`. `E501` is intentionally off
   (formatter owns line length).
4. **Tests: `pytest` + `hypothesis`, coverage ≥ 90%.** TDD for new behavior. The math is pure
   functions — prefer property tests. See [testing.md](docs/development/testing.md).
5. **Models emit their own pre-activations — NO `nn.Module` hooks.** This is a deliberate
   design decision; see
   [architecture/overview.md](docs/architecture/overview.md#key-decision-models-emit-their-own-pre-activations-no-hooks).
6. **Pure functions for the math** (`regularization/`); side effects (I/O, plotting,
   checkpoints) live in `training/` / `analysis/`.
7. **Keep modules single-purpose** per the [module layout](docs/architecture/overview.md). A
   file outgrowing one responsibility is a signal to split it.
8. **Update docs in the same change** — especially the
   [experiment log](docs/research/README.md#experiment-log) after running experiments.

## Where to look

| You want… | Go to |
| --- | --- |
| the idea + math | [docs/research/README.md](docs/research/README.md) |
| to set up / run | [docs/development/setup.md](docs/development/setup.md) |
| tools & commands | [docs/development/tooling.md](docs/development/tooling.md) |
| test conventions | [docs/development/testing.md](docs/development/testing.md) |
| code structure | [docs/architecture/overview.md](docs/architecture/overview.md) |
| how to contribute | [docs/development/contributing.md](docs/development/contributing.md) |

## Gotchas

- `ty` is **beta**: diagnostics can change between releases; it's pinned for a reason.
- Python 3.14 currently resolves to **3.14.0rc3** via `uv`'s index; if a dependency lacks a
  3.14 wheel, fall back to `uv python pin 3.13` (everything works on the floor).
- The CPU `torch` index is set in `pyproject.toml`; a plain `uv sync` will **not** pull CUDA.
- Roadmap items aren't filed as GitHub issues yet — see
  [contributing.md](docs/development/contributing.md#roadmap--issues).
