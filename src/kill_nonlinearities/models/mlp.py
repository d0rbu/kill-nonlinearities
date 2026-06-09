"""ReLU MLP that emits its own pre-activations (spec §4.3).

The forward stays plain ReLU via SelectiveReLU sites — NO nn.Module hooks. Each
``pre_activations[i]`` is the literal input tensor to ``activations[i]`` (load-bearing
for the I2 bit-exactness chain).
"""

from torch import Tensor, nn

from kill_nonlinearities.config import ModelConfig
from kill_nonlinearities.models.activations import SelectiveReLU
from kill_nonlinearities.models.outputs import ForwardOutput


class ReLUMLP(nn.Module):
    """Flatten → (Linear → SelectiveReLU)* → Linear head; emits ForwardOutput.

    ``linears``/``activations`` are typed Python lists holding the very modules
    registered (and thus tracked by ``state_dict``/``.to``) via the parallel
    ``nn.ModuleList`` containers.
    """

    linears: list[nn.Linear]
    activations: list[SelectiveReLU]

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.linears = []
        self.activations = []
        self._linear_list = nn.ModuleList()
        self._activation_list = nn.ModuleList()
        in_features = config.input_dim
        for width in config.hidden_dims:
            linear = nn.Linear(in_features, width)
            activation = SelectiveReLU(width)
            self.linears.append(linear)
            self.activations.append(activation)
            self._linear_list.append(linear)
            self._activation_list.append(activation)
            in_features = width
        self.head = nn.Linear(in_features, config.output_dim)
        self.site_names = tuple(f"relu{i}" for i in range(len(config.hidden_dims)))

    def forward(self, x: Tensor) -> ForwardOutput:
        x = x.flatten(1)
        pre: list[Tensor] = []
        for linear, act in zip(self.linears, self.activations, strict=True):
            z = linear(x)
            pre.append(z)
            x = act(z)
        return ForwardOutput(self.head(x), tuple(pre), self.site_names)
