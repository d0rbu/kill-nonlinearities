"""Forward-pass transport struct emitted by ReLUMLP (spec §4.1)."""

from dataclasses import dataclass

from torch import Tensor


@dataclass(frozen=True, eq=False)
class ForwardOutput:
    """Transport struct only; identity eq/hash (eq=False).

    Tensors are not value-comparable, so tests compare fields with
    ``torch.equal``, never ``==`` on the whole object.
    """

    logits: Tensor
    pre_activations: tuple[Tensor, ...]
    site_names: tuple[str, ...]
