"""Unit tests for ForwardOutput (spec §4.1)."""

import dataclasses

import pytest
import torch

from kill_nonlinearities.models.outputs import ForwardOutput


def test_forward_output_is_frozen() -> None:
    """ForwardOutput is immutable: assigning a field raises FrozenInstanceError."""
    out = ForwardOutput(
        logits=torch.zeros(2, 3),
        pre_activations=(torch.zeros(2, 4),),
        site_names=("relu0",),
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        out.logits = torch.ones(2, 3)  # ty: ignore[invalid-assignment]


def test_forward_output_uses_identity_equality() -> None:
    """eq=False means equality/hash fall back to object identity, not field values."""
    logits = torch.zeros(2, 3)
    pre = (torch.zeros(2, 4),)
    names = ("relu0",)
    a = ForwardOutput(logits=logits, pre_activations=pre, site_names=names)
    b = ForwardOutput(logits=logits, pre_activations=pre, site_names=names)
    assert a == a
    assert a != b
    assert hash(a) != hash(b)


def test_forward_output_transports_fields_verbatim() -> None:
    """Fields are stored unchanged; tensors compared with torch.equal (not ==)."""
    logits = torch.randn(2, 3)
    pre0 = torch.randn(2, 4)
    pre1 = torch.randn(2, 5)
    out = ForwardOutput(
        logits=logits,
        pre_activations=(pre0, pre1),
        site_names=("relu0", "relu1"),
    )
    assert torch.equal(out.logits, logits)
    assert len(out.pre_activations) == 2
    assert torch.equal(out.pre_activations[0], pre0)
    assert torch.equal(out.pre_activations[1], pre1)
    assert out.site_names == ("relu0", "relu1")
