# Contributing

[← Development](README.md) · [← Documentation hub](../README.md)

Conventions for changing this repo, whether you're a human or a coding agent. Agents should
also read [`AGENTS.md`](../../AGENTS.md).

## Workflow

1. Branch off `main`.
2. Prefer **test-driven development** for the research code: write a failing test that
   encodes the property/behavior, then implement until it passes. The math is pure functions
   — lean on [property-based tests](testing.md#property-based-testing-with-hypothesis).
3. Keep changes small and focused; one logical change per PR.
4. Run **`just check`** (lint + type-check + test) until green.
5. Update docs in the same change — especially the
   [experiment log](../research/README.md#experiment-log) when you run experiments, and
   [architecture/overview.md](../architecture/overview.md) if you change module boundaries.

## Definition of done

A change is done when **all** of these hold:

- [ ] `just check` is green (ruff lint + format, `ty`, tests).
- [ ] Coverage stays **≥ 90%**.
- [ ] New code is type-annotated and `ty`-clean.
- [ ] Public functions have docstrings; the math references
      [the method](../research/README.md#the-method).
- [ ] Docs updated (and cross-links still resolve).
- [ ] No new `nn.Module` hooks — models
      [emit their pre-activations](../architecture/overview.md#key-decision-models-emit-their-own-pre-activations-no-hooks).

## Code style

- Formatting and linting are **not manual** — `ruff` decides. Run `just fmt`.
- Type everything. `from __future__ import annotations` is fine; prefer precise types
  (`Tensor` shapes in docstrings/comments where they aid the reader).
- Keep modules single-purpose (see the [module layout](../architecture/overview.md)). A file
  growing past one clear responsibility is a signal to split it.
- Pure functions for the math; side effects (I/O, plotting, checkpoints) live in
  `training/`/`analysis/`, not in `regularization/`.

## Commits

- Small, descriptive messages; imperative mood ("Add entropy loss", not "Added").
- Don't commit data, checkpoints, or run artifacts — they're [git-ignored](../../.gitignore).
  Keep `uv.lock` committed.
- Commit `pre-commit` hook updates (`uv run pre-commit autoupdate`) on their own.

## Roadmap & issues

The plan of record is the [roadmap](../research/README.md#roadmap). Roadmap items aren't filed
as GitHub issues yet; tracked work lives in that roadmap for now. Mirror the roadmap items as
issues (toy-MLP regularizer + analysis, small transformer, phase-2 surgery) and link them back
from the roadmap.

---

[← Development](README.md) · [Research / the method →](../research/README.md)
