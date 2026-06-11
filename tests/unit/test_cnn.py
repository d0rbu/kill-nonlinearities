"""Unit tests for ReLUCNN, build_model, and the cnn ModelConfig (conv spec 2026-06-10)."""

import pytest
import torch

from kill_nonlinearities.analysis.statistics import neuron_stats
from kill_nonlinearities.config import ModelConfig
from kill_nonlinearities.models import build_model
from kill_nonlinearities.models.activations import ActivationMode
from kill_nonlinearities.models.cnn import ReLUCNN
from kill_nonlinearities.models.mlp import ReLUMLP


def _cnn_config() -> ModelConfig:
    """Tiny CNN: 2x8x8 input, conv channels (3, 4), one FC site of 6, 3 classes.

    Per-position neurons: conv0 has 3*8*8 = 192, conv1 has 4*4*4 = 64, fc0 has 6.
    """
    return ModelConfig(
        input_dim=1,  # ignored for kind="cnn" (head fan-in is derived)
        hidden_dims=(6,),
        output_dim=3,
        kind="cnn",
        in_channels=2,
        image_size=8,
        conv_channels=(3, 4),
    )


def _input(batch: int = 2) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(batch, 2, 8, 8)


def test_build_model_dispatches_on_kind() -> None:
    """build_model returns ReLUMLP for kind='mlp' and ReLUCNN for kind='cnn'."""
    mlp = build_model(ModelConfig(input_dim=12, hidden_dims=(4,), output_dim=3))
    cnn = build_model(_cnn_config())
    assert isinstance(mlp, ReLUMLP)
    assert isinstance(cnn, ReLUCNN)


def test_config_rejects_unknown_kind() -> None:
    with pytest.raises(ValueError, match="kind must be"):
        ModelConfig(input_dim=12, kind="transformer")


def test_config_cnn_requires_conv_channels() -> None:
    with pytest.raises(ValueError, match="non-empty conv_channels"):
        ModelConfig(input_dim=1, kind="cnn", conv_channels=())


def test_config_cnn_requires_divisible_image_size() -> None:
    """image_size must survive one 2x2 max-pool per conv block."""
    with pytest.raises(ValueError, match="divisible"):
        ModelConfig(input_dim=1, kind="cnn", image_size=6, conv_channels=(3, 4))


def test_site_names_and_activation_widths() -> None:
    """Sites are conv0..convN then fc0..; conv widths are C*H*W (per-position)."""
    model = ReLUCNN(_cnn_config())
    assert model.site_names == ("conv0", "conv1", "fc0")
    widths = [int(act.mode.shape[0]) for act in model.activations]
    assert widths == [3 * 8 * 8, 4 * 4 * 4, 6]


def test_site_group_sizes_are_positions_per_channel() -> None:
    """Conv sites group H*W positions per channel; MLP/fc sites group singly."""
    cnn = ReLUCNN(_cnn_config())
    assert cnn.site_group_sizes == (8 * 8, 4 * 4, 1)
    mlp = ReLUMLP(ModelConfig(input_dim=12, hidden_dims=(8, 6), output_dim=3))
    assert mlp.site_group_sizes == (1, 1)


def test_forward_shapes_logits_and_per_site_pre_activations() -> None:
    """Conv sites emit [B, C*H*W] (batch rows only); FC sites emit [B, N]."""
    model = ReLUCNN(_cnn_config())
    out = model(_input(batch=2))
    assert out.logits.shape == (2, 3)
    assert out.site_names == ("conv0", "conv1", "fc0")
    # conv0 sees 8x8 spatial, conv1 sees 4x4 (after one pool), fc0 is plain [B, N].
    assert out.pre_activations[0].shape == (2, 3 * 8 * 8)
    assert out.pre_activations[1].shape == (2, 4 * 4 * 4)
    assert out.pre_activations[2].shape == (2, 6)


def test_conv_emission_uses_flatten_index_convention() -> None:
    """Emitted neuron i = (c*H + h)*W + w matches z[b, c, h, w] exactly."""
    model = ReLUCNN(_cnn_config())
    model.eval()
    x = _input(batch=2)
    with torch.no_grad():
        out = model(x)
        z = model.convs[0](x)
    torch.testing.assert_close(out.pre_activations[0], z.flatten(1), rtol=0, atol=0)
    c, h, w = 1, 2, 5
    assert out.pre_activations[0][0, (c * 8 + h) * 8 + w] == z[0, c, h, w]


def test_all_relu_forward_matches_plain_relu_path_bitwise() -> None:
    """I1: the flattened SelectiveReLU path equals relu(conv(x)) bit-for-bit."""
    model = ReLUCNN(_cnn_config())
    model.eval()
    x = _input(batch=3)
    with torch.no_grad():
        logits = model(x).logits

        a = x
        for conv in model.convs:
            a = model.pool(torch.relu(conv(a)))
        a = a.flatten(1)
        for linear in model.linears:
            a = torch.relu(linear(a))
        ref = model.head(a)
    torch.testing.assert_close(logits, ref, rtol=0, atol=0)


def _channel_slice_modes(
    width: int, channel: int, positions: int, mode: ActivationMode
) -> torch.Tensor:
    """Full-width RELU modes with one channel's position block set to ``mode``."""
    modes = torch.zeros(width, dtype=torch.int64)
    modes[channel * positions : (channel + 1) * positions] = int(mode)
    return modes


def test_zero_mask_on_dead_positions_is_lossless() -> None:
    """Positions with q=0 (z<0 for the whole batch) mask to ZERO bit-exactly."""
    model = ReLUCNN(_cnn_config())
    model.eval()
    bias = model.convs[0].bias
    assert bias is not None
    with torch.no_grad():
        bias[1] = -1e3
    x = _input(batch=3)
    with torch.no_grad():
        baseline = model(x)
        # Channel 1's 64 positions (indices 64..128) really are dead on this batch.
        assert (baseline.pre_activations[0][:, 64:128] < 0).all()
        model.activations[0].set_modes(
            _channel_slice_modes(192, 1, 64, ActivationMode.ZERO)
        )
        masked = model(x).logits
    torch.testing.assert_close(masked, baseline.logits, rtol=0, atol=0)


def test_identity_mask_on_always_on_positions_is_lossless() -> None:
    """Positions with q=1 (z>0 for the whole batch) mask to IDENTITY bit-exactly."""
    model = ReLUCNN(_cnn_config())
    model.eval()
    bias = model.convs[0].bias
    assert bias is not None
    with torch.no_grad():
        bias[2] = 1e3
    x = _input(batch=3)
    with torch.no_grad():
        baseline = model(x)
        assert (baseline.pre_activations[0][:, 128:192] > 0).all()
        model.activations[0].set_modes(
            _channel_slice_modes(192, 2, 64, ActivationMode.IDENTITY)
        )
        masked = model(x).logits
    torch.testing.assert_close(masked, baseline.logits, rtol=0, atol=0)


def test_per_position_q_is_exact_for_constant_sign_channels() -> None:
    """Zero conv weights + signed biases give exact q in {0, 1} at every position."""
    model = ReLUCNN(_cnn_config())
    conv0 = model.convs[0]
    assert conv0.bias is not None
    with torch.no_grad():
        conv0.weight.zero_()
        conv0.bias.copy_(torch.tensor([2.0, -3.0, 0.5]))
    model.eval()
    with torch.no_grad():
        out = model(_input(batch=4))
    stats = neuron_stats({"conv0": out.pre_activations[0]})
    assert len(stats) == 192
    # All 64 positions of each channel inherit that channel's constant bias sign.
    assert all(s.q == 1.0 for s in stats[0:64])
    assert all(s.q == 0.0 for s in stats[64:128])
    assert all(s.q == 1.0 for s in stats[128:192])
    assert stats[0].mean_pre == pytest.approx(2.0)
    assert stats[64].mean_pre == pytest.approx(-3.0)
    assert stats[64].entropy == 0.0


def test_state_dict_round_trips_through_a_fresh_model() -> None:
    """A fresh ReLUCNN strict-loads the state dict and reproduces outputs exactly."""
    source = ReLUCNN(_cnn_config())
    clone = ReLUCNN(_cnn_config())
    clone.load_state_dict(source.state_dict())
    source.eval()
    clone.eval()
    x = _input(batch=2)
    with torch.no_grad():
        torch.testing.assert_close(clone(x).logits, source(x).logits, rtol=0, atol=0)
