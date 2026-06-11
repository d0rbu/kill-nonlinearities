"""Shared base for models that emit their own pre-activations (spec §4.3; conv spec 2026-06-10).

The contract (NO ``nn.Module`` hooks, per the architecture decision): ``forward``
returns a ``ForwardOutput`` whose ``pre_activations[i]`` is a 2-D ``[M, N_i]``
tensor for site ``i`` — rows are batch-like samples (for conv sites, batch x
spatial positions with channels last), columns are that site's neurons.
``activations[i]`` is the live ``SelectiveReLU`` applied at ``site_names[i]``.
"""

from torch import Tensor, nn

from kill_nonlinearities.models.activations import SelectiveReLU
from kill_nonlinearities.models.outputs import ForwardOutput


class PreActModel(nn.Module):
    """Base class for ``ReLUMLP``/``ReLUCNN``; trainer/analysis/surgery accept this.

    ``site_group_sizes[i]`` is the number of contiguous neurons per
    regularization *group* (channel) at site ``i`` — ``1`` means every neuron is
    its own group (MLP/fc sites); conv sites use ``H*W`` so one group spans a
    channel's positions. Consumed only by the channel-pooled regularizer
    variant (``RegConfig.granularity == "channel"``).
    """

    site_names: tuple[str, ...]
    site_group_sizes: tuple[int, ...]

    @property
    def activations(self) -> list[SelectiveReLU]:
        """The live ``SelectiveReLU`` modules, ordered like ``site_names``."""
        raise NotImplementedError

    def forward(self, x: Tensor) -> ForwardOutput:
        raise NotImplementedError
