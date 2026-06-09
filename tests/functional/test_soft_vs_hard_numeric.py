"""Functional numeric check: soft p_i and hard q_i on a fixed tensor (spec §4.4, §7)."""

import math

import torch

from kill_nonlinearities.regularization.entropy import (
    batch_fraction_positive,
    hard_fraction_positive,
)
from kill_nonlinearities.regularization.surrogate import soft_sign


def test_soft_and_hard_fraction_positive_match_hand_computed() -> None:
    """On a fixed [B, N] tensor, hard q_i and soft p_i equal hand-computed values."""
    # Two neurons, four samples.
    #   neuron 0 pre-acts: ( 2.0, -1.0,  3.0, -4.0)  -> strict z>0 fraction = 2/4 = 0.5
    #   neuron 1 pre-acts: ( 1.0,  1.0,  1.0,  0.0)  -> strict z>0 fraction = 3/4 = 0.75
    z = torch.tensor(
        [
            [2.0, 1.0],
            [-1.0, 1.0],
            [3.0, 1.0],
            [-4.0, 0.0],
        ]
    )
    tau = 0.5

    # Hard q_i (strict '>'): neuron 1 has a 0.0 entry which counts as NOT positive.
    hard_q = hard_fraction_positive(z)
    expected_hard = torch.tensor([0.5, 0.75])
    torch.testing.assert_close(hard_q, expected_hard, rtol=0, atol=1e-7)

    # Soft p_i = mean over batch of sigmoid(z / tau), computed by hand per element.
    def sig(v: float) -> float:
        return 1.0 / (1.0 + math.exp(-v / tau))

    expected_soft = torch.tensor(
        [
            (sig(2.0) + sig(-1.0) + sig(3.0) + sig(-4.0)) / 4.0,
            (sig(1.0) + sig(1.0) + sig(1.0) + sig(0.0)) / 4.0,
        ]
    )

    soft_p = batch_fraction_positive(z, tau)
    torch.testing.assert_close(soft_p, expected_soft, rtol=0, atol=1e-6)

    # And the underlying surrogate matches: mean over dim 0 of soft_sign(z, tau).
    soft_p_direct = soft_sign(z, tau).mean(0)
    torch.testing.assert_close(soft_p_direct, expected_soft, rtol=0, atol=1e-6)
