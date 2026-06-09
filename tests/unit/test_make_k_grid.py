"""Unit tests for analysis.selection.make_k_grid (spec §4.10, §7)."""

import pytest

from kill_nonlinearities.analysis.selection import make_k_grid


def test_make_k_grid_includes_zero_and_total() -> None:
    grid = make_k_grid(total=100, num_k=21)
    assert grid[0] == 0
    assert grid[-1] == 100


def test_make_k_grid_is_sorted_unique() -> None:
    grid = make_k_grid(total=100, num_k=21)
    assert grid == sorted(set(grid))


def test_make_k_grid_matches_rounding_formula() -> None:
    total, num_k = 100, 21
    expected = sorted({round(i * total / (num_k - 1)) for i in range(num_k)})
    assert make_k_grid(total, num_k) == expected


def test_make_k_grid_tiny_model_returns_unique_points_not_num_k() -> None:
    # total=4, num_k=21 collapses to the 5 unique points {0,1,2,3,4}.
    grid = make_k_grid(total=4, num_k=21)
    assert set(grid) == {0, 1, 2, 3, 4}
    assert grid == sorted(set(grid))
    assert 0 in grid
    assert 4 in grid
    # Assert membership/uniqueness, NOT len == num_k.
    assert len(grid) == 5


def test_make_k_grid_rejects_num_k_below_two() -> None:
    """num_k < 2 raises (the spacing divides by num_k - 1)."""
    with pytest.raises(ValueError, match="num_k"):
        make_k_grid(total=100, num_k=1)
