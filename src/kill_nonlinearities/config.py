"""Frozen configuration dataclasses (spec §3).

All dataclasses are ``frozen=True`` and have defaults; experiments override fields.
Only ``TempScheduleConfig`` carries validation (strict ``tau>0``, spec §3 [R16]).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class TempScheduleConfig:
    """Temperature-annealing schedule configuration (spec §3, §4.5)."""

    kind: str = "exponential"
    tau_start: float = 1.0
    tau_end: float = 0.1

    def __post_init__(self) -> None:
        if self.tau_start <= 0.0:
            raise ValueError(f"tau_start must be > 0, got {self.tau_start}")
        if self.tau_end <= 0.0:
            raise ValueError(f"tau_end must be > 0, got {self.tau_end}")
