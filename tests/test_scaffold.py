"""Scaffold smoke tests proving the toolchain (pytest + Hypothesis + coverage) runs.

These tests have no connection to the research idea; they exist only to verify the
test harness is wired up. Replace them with real tests as the library grows.
See docs/development/testing.md.
"""

from hypothesis import given
from hypothesis import strategies as st

import kill_nonlinearities


def test_package_exposes_version() -> None:
    """The package imports and exposes a non-empty string ``__version__``."""
    assert isinstance(kill_nonlinearities.__version__, str)
    assert kill_nonlinearities.__version__


@given(st.lists(st.integers()))
def test_sorting_is_idempotent(values: list[int]) -> None:
    """Scaffold property test: sorting an already-sorted list is a no-op."""
    once = sorted(values)
    twice = sorted(once)
    assert once == twice
