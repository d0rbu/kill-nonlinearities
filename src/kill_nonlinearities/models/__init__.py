"""Models: the pre-activation contract, ReLU MLP/CNN, and the config factory."""

from kill_nonlinearities.config import ModelConfig
from kill_nonlinearities.models.base import PreActModel
from kill_nonlinearities.models.cnn import ReLUCNN
from kill_nonlinearities.models.mlp import ReLUMLP

__all__ = ["PreActModel", "ReLUCNN", "ReLUMLP", "build_model"]


def build_model(config: ModelConfig) -> PreActModel:
    """Construct the configured model (``ModelConfig.kind``: ``mlp`` or ``cnn``)."""
    if config.kind == "cnn":
        return ReLUCNN(config)
    return ReLUMLP(config)
