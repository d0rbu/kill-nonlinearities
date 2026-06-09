"""ReLU MLP that emits its own pre-activations (spec §4.3).

The forward stays plain ReLU via SelectiveReLU sites — NO nn.Module hooks. Each
``pre_activations[i]`` is the literal input tensor to ``activations[i]`` (load-bearing
for the I2 bit-exactness chain).
"""

from typing import cast

from torch import Tensor, nn

from kill_nonlinearities.config import ModelConfig
from kill_nonlinearities.models.activations import SelectiveReLU
from kill_nonlinearities.models.outputs import ForwardOutput


class ReLUMLP(nn.Module):
    """Flatten → (Linear → SelectiveReLU)* → Linear head; emits ForwardOutput.

    The ``nn.ModuleList`` containers ``_linears``/``_activations`` are the single
    source of truth: they own registration, ``state_dict``, and ``.to``. The
    ``linears``/``activations`` properties surface those same live modules with
    precise element types for callers (e.g. ``activations[i].set_modes(...)``).
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self._linears = nn.ModuleList()
        self._activations = nn.ModuleList()
        in_features = config.input_dim
        for width in config.hidden_dims:
            self._linears.append(nn.Linear(in_features, width))
            self._activations.append(SelectiveReLU(width))
            in_features = width
        self.head = nn.Linear(in_features, config.output_dim)
        self.site_names = tuple(f"relu{i}" for i in range(len(config.hidden_dims)))

    # ``ty`` types ``nn.ModuleList[int] -> Module``, erasing element types; these
    # accessors recover them while returning the live, registered modules.
    @property
    def linears(self) -> list[nn.Linear]:
        return cast(list[nn.Linear], list(self._linears))

    @property
    def activations(self) -> list[SelectiveReLU]:
        return cast(list[SelectiveReLU], list(self._activations))

    def forward(self, x: Tensor) -> ForwardOutput:
        x = x.flatten(1)
        pre: list[Tensor] = []
        for linear, act in zip(self.linears, self.activations, strict=True):
            z = linear(x)
            pre.append(z)
            x = act(z)
        return ForwardOutput(self.head(x), tuple(pre), self.site_names)
