"""Unit tests for the frozen config dataclasses (spec §3)."""

import dataclasses

import pytest

from kill_nonlinearities.config import TempScheduleConfig


def test_temp_schedule_config_is_frozen() -> None:
    """TempScheduleConfig is a frozen dataclass with the documented defaults."""
    cfg = TempScheduleConfig()
    assert cfg.kind == "exponential"
    assert cfg.tau_start == 1.0
    assert cfg.tau_end == 0.1
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.tau_start = 2.0  # ty: ignore[invalid-assignment]


def test_temp_schedule_config_rejects_non_positive_tau_start() -> None:
    """tau_start must be strictly positive (spec §3 [R16])."""
    with pytest.raises(ValueError, match="tau_start"):
        TempScheduleConfig(tau_start=0.0)


def test_temp_schedule_config_rejects_non_positive_tau_end() -> None:
    """tau_end must be strictly positive (spec §3 [R16])."""
    with pytest.raises(ValueError, match="tau_end"):
        TempScheduleConfig(tau_end=-0.5)
