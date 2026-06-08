# Architecture

[← Documentation hub](../README.md)

How the code is *planned* to be organized, and the design decisions behind it. The codebase
is currently a scaffold, so this section is mostly forward-looking — it records intent so
that implementation stays coherent.

## Contents

- **[overview.md](overview.md)** — planned module layout and the key design decisions
  (notably: models emit their own pre-activations rather than relying on `nn.Module` hooks).

## Relationship to the rest of the docs

- The *what* and *why* (the method) live in [research/](../research/README.md).
- The *how to build/test* lives in [development/](../development/README.md).
- This section is the *how the code is shaped*.

---

[← Documentation hub](../README.md) · [Architecture overview →](overview.md)
