"""Unit tests for regularization.surrogate (spec §4.4, §7)."""

import importlib


def test_regularization_package_importable() -> None:
    """The regularization subpackage imports cleanly."""
    module = importlib.import_module("kill_nonlinearities.regularization")
    assert module is not None
