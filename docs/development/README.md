# Development

[← Documentation hub](../README.md)

Everything you need to work in this repository.

## Contents

| Doc | Covers |
| --- | --- |
| **[setup.md](setup.md)** | Installing `uv`/`just`, the Python version, creating the environment, CPU vs. CUDA `torch`. |
| **[tooling.md](tooling.md)** | `uv`, `ruff`, `ty`, `pre-commit`, and every `just` recipe. |
| **[testing.md](testing.md)** | `pytest`, `hypothesis`, coverage conventions. |
| **[contributing.md](contributing.md)** | Workflow, the definition of done, commits, code style, filing roadmap issues. |

## TL;DR

```bash
just setup     # create the env + install git hooks
just check     # lint + type-check + test (run this before every commit)
```

If you only read one more page, read [tooling.md](tooling.md).

## Conventions at a glance

- **`uv` for everything** — never `pip install` into the environment; use `uv add` / `uv add --dev`.
- **`ty` must pass** and new code is type-annotated. **`ruff`** owns lint + formatting.
- **Tests** use `pytest` + `hypothesis`; coverage gate is **≥ 90%**.
- `just check` is the local gate; `pre-commit` enforces a subset on every commit.
- Models **emit their own pre-activations** — no `nn.Module` hooks. See
  [architecture/overview.md](../architecture/overview.md).

---

[← Documentation hub](../README.md) · [Setup →](setup.md)
