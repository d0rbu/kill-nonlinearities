"""Per-neuron masked activation for surgery (spec §4.2)."""

from enum import IntEnum

import torch
from torch import Tensor, nn


class ActivationMode(IntEnum):
    """Per-neuron activation selector; integer codes are frozen by the spec."""

    RELU = 0
    ZERO = 1
    IDENTITY = 2


class SelectiveReLU(nn.Module):
    """ReLU whose per-neuron behavior is switched by an int64 ``mode`` buffer.

    Modes broadcast a ``[N]`` selector over a ``[B, N]`` batch. All-``RELU`` (the
    default) reproduces ``torch.relu`` bit-for-bit (invariant I1). The buffer moves
    with ``.to(device)`` and round-trips through ``state_dict``; it is never trained.
    """

    mode: Tensor

    def __init__(self, num_features: int) -> None:
        super().__init__()
        self.num_features = num_features
        self.register_buffer("mode", torch.zeros(num_features, dtype=torch.int64))

    def forward(self, z: Tensor) -> Tensor:
        relu = torch.relu(z)
        out = torch.where(self.mode == int(ActivationMode.RELU), relu, z)
        return torch.where(
            self.mode == int(ActivationMode.ZERO), torch.zeros_like(z), out
        )

    def set_modes(self, modes: Tensor) -> None:
        self.mode.copy_(modes)

    def reset(self) -> None:
        self.mode.zero_()
