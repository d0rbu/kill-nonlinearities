"""Unit tests for ActivationMode + SelectiveReLU (spec §4.2, §6 I1)."""

import pytest
import torch

from kill_nonlinearities.models.activations import ActivationMode, SelectiveReLU


def test_activation_mode_int_values() -> None:
    """The enum's integer codes are frozen by the spec: RELU=0, ZERO=1, IDENTITY=2."""
    assert int(ActivationMode.RELU) == 0
    assert int(ActivationMode.ZERO) == 1
    assert int(ActivationMode.IDENTITY) == 2


def test_default_mode_buffer_is_all_relu_int64() -> None:
    """Default mode buffer is an int64 [N] tensor of RELU (0), registered as a buffer."""
    act = SelectiveReLU(num_features=5)
    assert act.mode.dtype == torch.int64
    assert act.mode.shape == (5,)
    assert torch.equal(act.mode, torch.zeros(5, dtype=torch.int64))
    assert "mode" in dict(act.named_buffers())


def test_all_relu_equals_torch_relu_bit_for_bit() -> None:
    """I1: default (all-RELU) SelectiveReLU output equals torch.relu exactly."""
    act = SelectiveReLU(num_features=4)
    z = torch.tensor([[-2.0, -1e-9, 0.0, 3.0], [1.5, -0.5, 7.0, -8.0]])
    assert torch.equal(act(z), torch.relu(z))


def test_zero_mode_outputs_zeros() -> None:
    """A ZERO neuron emits 0 regardless of the pre-activation."""
    act = SelectiveReLU(num_features=2)
    act.set_modes(torch.tensor([int(ActivationMode.ZERO), int(ActivationMode.RELU)]))
    z = torch.tensor([[5.0, 5.0], [-3.0, -3.0]])
    out = act(z)
    assert torch.equal(out[:, 0], torch.zeros(2))
    assert torch.equal(out[:, 1], torch.relu(z[:, 1]))


def test_identity_mode_passes_through_exactly() -> None:
    """An IDENTITY neuron emits z unchanged, including negative values."""
    act = SelectiveReLU(num_features=2)
    act.set_modes(
        torch.tensor([int(ActivationMode.IDENTITY), int(ActivationMode.RELU)])
    )
    z = torch.tensor([[-4.0, -4.0], [2.0, 2.0]])
    out = act(z)
    assert torch.equal(out[:, 0], z[:, 0])
    assert torch.equal(out[:, 1], torch.relu(z[:, 1]))


def test_mixed_modes_per_neuron() -> None:
    """RELU/ZERO/IDENTITY can coexist per neuron in one call."""
    act = SelectiveReLU(num_features=3)
    act.set_modes(
        torch.tensor(
            [
                int(ActivationMode.RELU),
                int(ActivationMode.ZERO),
                int(ActivationMode.IDENTITY),
            ]
        )
    )
    z = torch.tensor([[-1.0, 9.0, -2.0], [4.0, 9.0, -2.0]])
    expected = torch.tensor([[0.0, 0.0, -2.0], [4.0, 0.0, -2.0]])
    assert torch.equal(act(z), expected)


def test_mode_broadcasts_over_batch() -> None:
    """The [N] mode broadcasts over a [B, N] batch."""
    act = SelectiveReLU(num_features=3)
    act.set_modes(
        torch.tensor(
            [
                int(ActivationMode.ZERO),
                int(ActivationMode.IDENTITY),
                int(ActivationMode.RELU),
            ]
        )
    )
    z = torch.randn(7, 3)
    out = act(z)
    assert out.shape == (7, 3)
    assert torch.equal(out[:, 0], torch.zeros(7))
    assert torch.equal(out[:, 1], z[:, 1])
    assert torch.equal(out[:, 2], torch.relu(z[:, 2]))


def test_reset_restores_all_relu() -> None:
    """reset() returns every neuron to RELU."""
    act = SelectiveReLU(num_features=3)
    act.set_modes(torch.tensor([1, 2, 1], dtype=torch.int64))
    act.reset()
    assert torch.equal(act.mode, torch.zeros(3, dtype=torch.int64))


def test_set_modes_accepts_valid_full_width_int64() -> None:
    """A valid [N] int64 tensor with values in {0,1,2} is accepted and stored."""
    act = SelectiveReLU(num_features=3)
    modes = torch.tensor([0, 1, 2], dtype=torch.int64)
    act.set_modes(modes)
    assert torch.equal(act.mode, modes)


def test_set_modes_rejects_wrong_shape() -> None:
    """A modes tensor that is not exactly (num_features,) raises ValueError."""
    act = SelectiveReLU(num_features=3)
    with pytest.raises(ValueError, match="shape"):
        act.set_modes(torch.zeros(4, dtype=torch.int64))


def test_set_modes_rejects_wrong_dtype() -> None:
    """A non-int64 modes tensor raises ValueError."""
    act = SelectiveReLU(num_features=3)
    with pytest.raises(ValueError, match="int64"):
        act.set_modes(torch.zeros(3, dtype=torch.int32))


def test_set_modes_rejects_out_of_range_values() -> None:
    """Values outside {0, 1, 2} raise ValueError."""
    act = SelectiveReLU(num_features=3)
    with pytest.raises(ValueError, match="0, 1, 2"):
        act.set_modes(torch.tensor([0, 3, 1], dtype=torch.int64))
    with pytest.raises(ValueError, match="0, 1, 2"):
        act.set_modes(torch.tensor([-1, 0, 1], dtype=torch.int64))


def test_gradient_routing_is_exact_per_mode() -> None:
    """Backward routes grads exactly: ZERO→0, IDENTITY→1, RELU→(z>0).float()."""
    act = SelectiveReLU(num_features=3)
    act.set_modes(
        torch.tensor(
            [
                int(ActivationMode.RELU),
                int(ActivationMode.ZERO),
                int(ActivationMode.IDENTITY),
            ]
        )
    )
    z = torch.tensor([[-1.0, 5.0, -2.0], [3.0, 5.0, 4.0]], requires_grad=True)
    out = act(z)
    out.sum().backward()
    assert z.grad is not None
    expected = torch.tensor(
        [
            [(z[0, 0] > 0).float(), 0.0, 1.0],
            [(z[1, 0] > 0).float(), 0.0, 1.0],
        ]
    )
    assert torch.equal(z.grad, expected)


def test_relu_subgradient_at_zero_is_zero() -> None:
    """RELU's subgradient at z==0 is 0 (relu convention), asserted exactly."""
    act = SelectiveReLU(num_features=1)
    z = torch.zeros(2, 1, requires_grad=True)
    act(z).sum().backward()
    assert z.grad is not None
    assert torch.equal(z.grad, torch.zeros(2, 1))


def test_state_dict_round_trip_preserves_non_default_mode() -> None:
    """A non-default mode buffer survives a strict state_dict save/load round-trip."""
    src = SelectiveReLU(num_features=3)
    src.set_modes(torch.tensor([2, 1, 0], dtype=torch.int64))
    dst = SelectiveReLU(num_features=3)
    dst.load_state_dict(src.state_dict())
    assert dst.mode.dtype == torch.int64
    assert torch.equal(dst.mode, torch.tensor([2, 1, 0], dtype=torch.int64))
    z = torch.tensor([[-1.0, 5.0, -2.0]])
    assert torch.equal(dst(z), src(z))


def test_to_float64_keeps_mode_comparisons_working() -> None:
    """After .to(torch.float64) the int64 mode buffer is unchanged and forward works."""
    act = SelectiveReLU(num_features=3).to(torch.float64)
    act.set_modes(
        torch.tensor(
            [
                int(ActivationMode.ZERO),
                int(ActivationMode.IDENTITY),
                int(ActivationMode.RELU),
            ]
        )
    )
    assert act.mode.dtype == torch.int64
    z = torch.tensor([[-1.0, -2.0, -3.0], [4.0, 5.0, 6.0]], dtype=torch.float64)
    out = act(z)
    assert out.dtype == torch.float64
    assert torch.equal(out[:, 0], torch.zeros(2, dtype=torch.float64))
    assert torch.equal(out[:, 1], z[:, 1])
    assert torch.equal(out[:, 2], torch.relu(z[:, 2]))
