# Documentation

[← Repository root](../README.md)

Documentation for **kill-nonlinearities** — a project testing whether a regularizer can
push a network's ReLUs toward **sign-consistent** pre-activations, so that nonlinearities
which never actually switch can be identified and removed. For the one-paragraph pitch see
the [root README](../README.md); for the full idea and math see
[research](research/README.md).

## Map

This `docs/` tree is hierarchical — every directory has a `README.md` that links to its
contents and back here.

| Section | What's in it |
| --- | --- |
| 🔬 **[research/](research/README.md)** | The canonical method writeup (motivation, the math, the elimination plan) **and** the running experiment log. Start here for *what* we're doing and *why*. |
| 🛠️ **[development/](development/README.md)** | How to work in the repo: [setup](development/setup.md), [tooling](development/tooling.md), [testing](development/testing.md), [contributing](development/contributing.md). |
| 🏗️ **[architecture/](architecture/README.md)** | The *planned* code structure and key design decisions (e.g. models emit their own pre-activations instead of using hooks). Mostly future-facing; the code is a scaffold today. |

## Where should I look?

- *"What is this project and what's the math?"* → [research/README.md](research/README.md)
- *"How do I set up and run things?"* → [development/setup.md](development/setup.md)
- *"What are the tools and commands?"* → [development/tooling.md](development/tooling.md)
- *"How are tests written here?"* → [development/testing.md](development/testing.md)
- *"How will the code be organized?"* → [architecture/overview.md](architecture/overview.md)
- *"I'm a coding agent — what are the rules?"* → [AGENTS.md](../AGENTS.md)

## Project status

🚧 **Scaffold.** Tooling, the test harness, and these docs are in place; no research code is
implemented yet. The plan of record is in the [roadmap](research/README.md#roadmap).
