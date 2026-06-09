"""Unit tests for ``TemperatureSchedule`` (spec §4.5)."""

import itertools
import math

import pytest

from kill_nonlinearities.training.schedule import TemperatureSchedule


def test_constant_returns_tau_start_everywhere() -> None:
    sched = TemperatureSchedule(
        kind="constant", tau_start=0.7, tau_end=0.1, total_steps=10
    )
    assert all(sched(step) == 0.7 for step in range(10))


@pytest.mark.parametrize("kind", ["exponential", "linear"])
def test_endpoints_hit_tau_start_and_tau_end(kind: str) -> None:
    sched = TemperatureSchedule(kind=kind, tau_start=1.0, tau_end=0.1, total_steps=11)
    assert sched(0) == pytest.approx(1.0)
    assert sched(10) == pytest.approx(0.1)


def test_exponential_is_geometric_midpoint() -> None:
    sched = TemperatureSchedule(
        kind="exponential", tau_start=1.0, tau_end=0.01, total_steps=3
    )
    # geometric interpolation: tau(1) == sqrt(tau_start * tau_end)
    assert sched(1) == pytest.approx(math.sqrt(1.0 * 0.01))


def test_linear_is_arithmetic_midpoint() -> None:
    sched = TemperatureSchedule(
        kind="linear", tau_start=1.0, tau_end=0.0 + 0.2, total_steps=3
    )
    assert sched(1) == pytest.approx((1.0 + 0.2) / 2.0)


@pytest.mark.parametrize("kind", ["exponential", "linear"])
def test_monotone_decreasing_when_tau_end_below_tau_start(kind: str) -> None:
    sched = TemperatureSchedule(kind=kind, tau_start=1.0, tau_end=0.1, total_steps=8)
    values = [sched(step) for step in range(8)]
    assert all(later <= earlier for earlier, later in itertools.pairwise(values))


@pytest.mark.parametrize("kind", ["exponential", "linear", "constant"])
@pytest.mark.parametrize("total_steps", [0, 1])
def test_total_steps_le_one_returns_tau_start_no_div_by_zero(
    kind: str, total_steps: int
) -> None:
    sched = TemperatureSchedule(
        kind=kind, tau_start=0.5, tau_end=0.1, total_steps=total_steps
    )
    assert sched(0) == 0.5
    assert sched(5) == 0.5  # step beyond range still safe


@pytest.mark.parametrize("kind", ["exponential", "linear"])
def test_step_beyond_total_clamps_to_last_value(kind: str) -> None:
    sched = TemperatureSchedule(kind=kind, tau_start=1.0, tau_end=0.1, total_steps=5)
    assert sched(100) == pytest.approx(sched(4))


@pytest.mark.parametrize("kind", ["exponential", "linear", "constant"])
def test_tau_is_positive_for_every_step(kind: str) -> None:
    sched = TemperatureSchedule(kind=kind, tau_start=2.0, tau_end=0.001, total_steps=50)
    assert all(sched(step) > 0.0 for step in range(50))


@pytest.mark.parametrize(
    ("tau_start", "tau_end"),
    [(0.0, 0.1), (-1.0, 0.1), (1.0, 0.0), (1.0, -0.1)],
)
def test_non_positive_tau_raises(tau_start: float, tau_end: float) -> None:
    with pytest.raises(ValueError, match="tau"):
        TemperatureSchedule(
            kind="exponential",
            tau_start=tau_start,
            tau_end=tau_end,
            total_steps=10,
        )


def test_unknown_kind_raises() -> None:
    with pytest.raises(ValueError, match="kind"):
        TemperatureSchedule(kind="cosine", tau_start=1.0, tau_end=0.1, total_steps=10)
