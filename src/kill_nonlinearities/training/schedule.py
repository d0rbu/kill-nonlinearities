"""Temperature annealing schedule and checkpoint-step enumeration (spec §4.5)."""


class TemperatureSchedule:
    """Map a training ``step`` to a temperature ``tau``.

    ``tau(0) == tau_start``; for the non-constant kinds ``tau(total_steps - 1)
    == tau_end`` (``exponential`` = geometric interpolation, ``linear`` =
    arithmetic interpolation). ``constant`` ignores ``tau_end``. When
    ``total_steps <= 1`` the schedule is pinned to ``tau_start`` (the
    ``total_steps - 1`` denominator is guarded and the endpoint contract is
    waived). Steps at or beyond ``total_steps`` clamp to the last value.
    """

    _KINDS = ("exponential", "linear", "constant")

    def __init__(
        self, kind: str, tau_start: float, tau_end: float, total_steps: int
    ) -> None:
        if kind not in self._KINDS:
            msg = f"unknown schedule kind {kind!r}; expected one of {self._KINDS}"
            raise ValueError(msg)
        if tau_start <= 0.0 or tau_end <= 0.0:
            msg = (
                "tau_start and tau_end must be strictly positive; "
                f"got tau_start={tau_start}, tau_end={tau_end}"
            )
            raise ValueError(msg)
        self.kind = kind
        self.tau_start = tau_start
        self.tau_end = tau_end
        self.total_steps = total_steps

    def __call__(self, step: int) -> float:
        if self.kind == "constant" or self.total_steps <= 1:
            return self.tau_start
        last = self.total_steps - 1
        clamped = step if step < last else last
        fraction = clamped / last
        if self.kind == "linear":
            return self.tau_start + (self.tau_end - self.tau_start) * fraction
        # exponential == geometric interpolation between the endpoints.
        return self.tau_start * (self.tau_end / self.tau_start) ** fraction


def checkpoint_steps(
    total_steps: int, steps_per_epoch: int, every_epochs: int
) -> list[int]:
    """Return the sorted, de-duplicated set of steps at which to checkpoint.

    Always includes step ``0`` and the final step ``total_steps - 1``, plus
    every multiple of ``every_epochs * steps_per_epoch`` that falls inside
    ``[0, total_steps)``.
    """
    if total_steps <= 0:
        return []
    stride = every_epochs * steps_per_epoch
    steps = {0, total_steps - 1}
    if stride > 0:
        steps.update(range(0, total_steps, stride))
    return sorted(steps)
