"""ReLU CNN that emits its own pre-activations (conv spec 2026-06-10).

Per-position neuron semantics: at a conv site a "neuron" is an individual output
position ``(c, h, w)``, statistically identical to an MLP neuron — its ``q_i`` is
sampled over the **batch only**. Each conv site flattens its pre-activations to
``[B, C*H*W]`` (index ``i = (c*H + h)*W + w``), applies its
``SelectiveReLU(C*H*W)`` on that view (``mode`` broadcasts over the last dim),
and reshapes back before pooling. ``torch.relu``/``torch.where`` are elementwise,
so the flattened path stays bit-identical to plain ``relu(conv(x))`` (invariant
I1), and every downstream consumer (regularizer, statistics, surgery, probes)
sees the standard ``[M, N]`` per-site contract unchanged.
"""

from typing import cast

from torch import Tensor, nn

from kill_nonlinearities.config import ModelConfig
from kill_nonlinearities.models.activations import SelectiveReLU
from kill_nonlinearities.models.base import PreActModel
from kill_nonlinearities.models.outputs import ForwardOutput


class ReLUCNN(PreActModel):
    """(Conv3x3 pad 1 → SelectiveReLU per position → MaxPool 2x2)* → MLP head.

    Site names are ``conv0..`` for the conv blocks then ``fc0..`` for the hidden
    linear layers; ``activations`` surfaces all sites in that order. Conv block
    ``i`` sees a ``image_size / 2^i`` square input (3x3/pad-1 convs preserve the
    side; each block pools once after activation), so its site has
    ``conv_channels[i] * (image_size / 2^i)^2`` neurons. Kernel, padding, and
    pooling are fixed (3 / 1 / 2) per the conv spec.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self._convs = nn.ModuleList()
        self._conv_acts = nn.ModuleList()
        conv_group_sizes: list[int] = []
        in_ch = config.in_channels
        side = config.image_size
        for ch in config.conv_channels:
            self._convs.append(nn.Conv2d(in_ch, ch, kernel_size=3, padding=1))
            self._conv_acts.append(SelectiveReLU(ch * side * side))
            conv_group_sizes.append(side * side)  # one group per channel
            in_ch = ch
            side //= 2
        self.pool = nn.MaxPool2d(2)

        in_features = in_ch * side * side
        self._linears = nn.ModuleList()
        self._fc_acts = nn.ModuleList()
        for width in config.hidden_dims:
            self._linears.append(nn.Linear(in_features, width))
            self._fc_acts.append(SelectiveReLU(width))
            in_features = width
        self.head = nn.Linear(in_features, config.output_dim)

        self.site_names = tuple(
            f"conv{i}" for i in range(len(config.conv_channels))
        ) + tuple(f"fc{i}" for i in range(len(config.hidden_dims)))
        self.site_group_sizes = tuple(conv_group_sizes) + tuple(
            1 for _ in config.hidden_dims
        )

    # ``ty`` types ``nn.ModuleList[int] -> Module``, erasing element types; these
    # accessors recover them while returning the live, registered modules.
    @property
    def convs(self) -> list[nn.Conv2d]:
        return cast(list[nn.Conv2d], list(self._convs))

    @property
    def linears(self) -> list[nn.Linear]:
        return cast(list[nn.Linear], list(self._linears))

    @property
    def activations(self) -> list[SelectiveReLU]:
        return cast(list[SelectiveReLU], list(self._conv_acts) + list(self._fc_acts))

    def forward(self, x: Tensor) -> ForwardOutput:
        pre: list[Tensor] = []
        for conv, act in zip(self._convs, self._conv_acts, strict=True):
            z = conv(x)
            z_flat = z.flatten(1)  # [B, C*H*W]: one neuron per (c, h, w)
            pre.append(z_flat)
            x = self.pool(act(z_flat).view(z.shape))
        x = x.flatten(1)
        for linear, act in zip(self._linears, self._fc_acts, strict=True):
            z = linear(x)
            pre.append(z)
            x = act(z)
        return ForwardOutput(self.head(x), tuple(pre), self.site_names)
