"""Functional tests for ReLUMLP (spec §4.3, §6 I1)."""

import torch

from kill_nonlinearities.config import ModelConfig
from kill_nonlinearities.models.activations import ActivationMode
from kill_nonlinearities.models.mlp import ReLUMLP
from kill_nonlinearities.models.outputs import ForwardOutput


def test_forward_returns_forward_output_with_expected_shapes() -> None:
    """forward emits a ForwardOutput: logits [B,C] and one pre-activation per site."""
    config = ModelConfig(input_dim=8, hidden_dims=(5, 3), output_dim=4)
    model = ReLUMLP(config)
    x = torch.randn(6, 8)
    out = model(x)
    assert isinstance(out, ForwardOutput)
    assert out.logits.shape == (6, 4)
    assert len(out.pre_activations) == 2
    assert out.pre_activations[0].shape == (6, 5)
    assert out.pre_activations[1].shape == (6, 3)


def test_site_names_are_stable_and_aligned() -> None:
    """site_names are ('relu0','relu1',...), one per hidden site, aligned to pre_acts."""
    config = ModelConfig(input_dim=8, hidden_dims=(5, 3), output_dim=4)
    model = ReLUMLP(config)
    assert model.site_names == ("relu0", "relu1")
    out = model(torch.randn(2, 8))
    assert out.site_names == ("relu0", "relu1")
    assert len(out.site_names) == len(out.pre_activations)


def test_forward_flattens_image_shaped_input() -> None:
    """flatten(1) collapses [B,1,28,28] (MNIST) to [B, input_dim] before first linear."""
    config = ModelConfig(input_dim=1 * 4 * 4, hidden_dims=(7,), output_dim=3)
    model = ReLUMLP(config)
    x = torch.randn(5, 1, 4, 4)
    out = model(x)
    assert out.logits.shape == (5, 3)
    assert out.pre_activations[0].shape == (5, 7)


def test_activations_are_selective_relu_per_hidden_site() -> None:
    """One SelectiveReLU per hidden site, each sized to that layer's width, default RELU."""
    config = ModelConfig(input_dim=8, hidden_dims=(5, 3), output_dim=4)
    model = ReLUMLP(config)
    assert len(model.activations) == 2
    assert model.activations[0].num_features == 5
    assert model.activations[1].num_features == 3
    assert int(model.activations[0].mode[0]) == int(ActivationMode.RELU)


def test_pre_activations_match_manual_layer_by_layer_recompute() -> None:
    """pre_activations[i] equals the literal input to activations[i] (bit-for-bit)."""
    torch.manual_seed(0)
    config = ModelConfig(input_dim=8, hidden_dims=(5, 3), output_dim=4)
    model = ReLUMLP(config)
    model.eval()
    x = torch.randn(6, 8)
    out = model(x)

    # Manual recompute using the model's own parameters and SelectiveReLU sites.
    h = x.flatten(1)
    z0 = model.linears[0](h)
    a0 = model.activations[0](z0)
    z1 = model.linears[1](a0)
    a1 = model.activations[1](z1)
    expected_logits = model.head(a1)

    assert torch.equal(out.pre_activations[0], z0)
    assert torch.equal(out.pre_activations[1], z1)
    assert torch.equal(out.logits, expected_logits)
