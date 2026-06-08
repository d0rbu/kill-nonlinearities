"""kill-nonlinearities: regularizing networks toward sign-consistent pre-activations.

The research idea, method, and math live in ``docs/research/README.md``; repository
conventions live in ``AGENTS.md``. This package is currently a scaffold — no research
code has been implemented yet.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("kill-nonlinearities")
except PackageNotFoundError:  # pragma: no cover - package not installed in env
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
