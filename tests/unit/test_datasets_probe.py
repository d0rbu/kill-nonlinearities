"""Unit tests for probe selection determinism (spec §4.9, §7 probe fixity)."""

import pytest
import torch

from kill_nonlinearities.config import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    ProbeConfig,
)
from kill_nonlinearities.data.datasets import (
    build_transform,
    make_dataloaders,
    make_probe_batch,
    select_probe_neurons,
)
from kill_nonlinearities.models.mlp import ReLUMLP


def _model() -> ReLUMLP:
    """A small ReLUMLP with two SelectiveReLU sites (hidden_dims=(8, 6))."""
    return ReLUMLP(ModelConfig(input_dim=12, hidden_dims=(8, 6), output_dim=3))


def test_select_probe_neurons_is_deterministic_under_seed() -> None:
    """Same model + seed → identical (site, idx) list (probe fixity)."""
    model = _model()
    a = select_probe_neurons(model, num=5, seed=0)
    b = select_probe_neurons(model, num=5, seed=0)
    assert a == b


def test_select_probe_neurons_returns_requested_count() -> None:
    """Returns exactly ``num`` distinct (site, idx) pairs."""
    model = _model()
    selected = select_probe_neurons(model, num=5, seed=0)
    assert len(selected) == 5
    assert len(set(selected)) == 5


def test_select_probe_neurons_indices_are_valid() -> None:
    """Every (site, idx) references a real site and an in-range neuron index."""
    model = _model()
    selected = select_probe_neurons(model, num=7, seed=1)
    widths = {
        name: int(act.mode.shape[0])
        for name, act in zip(model.site_names, model.activations, strict=True)
    }
    for site, idx in selected:
        assert site in widths
        assert 0 <= idx < widths[site]


def test_select_probe_neurons_changes_with_seed() -> None:
    """A different seed yields a different probe set (non-degenerate)."""
    model = _model()
    assert select_probe_neurons(model, num=5, seed=0) != select_probe_neurons(
        model, num=5, seed=1
    )


def test_select_probe_neurons_clamps_to_total_neurons() -> None:
    """Requesting more than total neurons returns all of them, no duplicates."""
    model = ReLUMLP(ModelConfig(input_dim=4, hidden_dims=(2, 2), output_dim=2))
    selected = select_probe_neurons(model, num=100, seed=0)
    total = 2 + 2
    assert len(selected) == total
    assert len(set(selected)) == total


def _probe_config() -> ExperimentConfig:
    """Synthetic config with a small probe batch (no network)."""
    return ExperimentConfig(
        name="probe-test",
        model=ModelConfig(input_dim=12, hidden_dims=(8, 6), output_dim=3),
        data=DataConfig(
            dataset="synthetic",
            batch_size=8,
            eval_batch_size=16,
            val_fraction=0.2,
            split_seed=0,
            num_workers=0,
        ),
        probe=ProbeConfig(num_neurons=4, seed=0, batch_size=10),
    )


def test_make_probe_batch_is_deterministic() -> None:
    """Same loader + size + seed → byte-identical probe batch (probe fixity)."""
    _, val, _ = make_dataloaders(_probe_config())
    a = make_probe_batch(val, size=10, seed=0)
    b = make_probe_batch(val, size=10, seed=0)
    assert torch.equal(a, b)


def test_make_probe_batch_shape_and_dtype() -> None:
    """Probe batch is [size, input_dim] float32 inputs only (no labels)."""
    config = _probe_config()
    _, val, _ = make_dataloaders(config)
    batch = make_probe_batch(val, size=10, seed=0)
    assert isinstance(batch, torch.Tensor)
    assert batch.shape == (10, config.model.input_dim)
    assert batch.dtype == torch.float32


def test_make_probe_batch_changes_with_seed() -> None:
    """A different seed selects a different (non-degenerate) probe batch."""
    _, val, _ = make_dataloaders(_probe_config())
    assert not torch.equal(
        make_probe_batch(val, size=10, seed=0),
        make_probe_batch(val, size=10, seed=1),
    )


def test_make_probe_batch_clamps_to_dataset_size() -> None:
    """Requesting more than the dataset holds returns all of it, no error."""
    _, val, _ = make_dataloaders(_probe_config())
    n = len(val.dataset)  # ty: ignore[invalid-argument-type]
    batch = make_probe_batch(val, size=n + 1000, seed=0)
    assert batch.shape[0] == n


def test_build_transform_mnist_has_no_augmentation() -> None:
    """MNIST transform is exactly ToTensor + Normalize (no random augmentation)."""
    transform = build_transform("mnist")
    names = [type(t).__name__ for t in transform.transforms]
    assert names == ["ToTensor", "Normalize"]


def test_build_transform_cifar10_has_no_augmentation() -> None:
    """CIFAR-10 transform is exactly ToTensor + Normalize (no random augmentation)."""
    transform = build_transform("cifar10")
    names = [type(t).__name__ for t in transform.transforms]
    assert names == ["ToTensor", "Normalize"]


def test_build_transform_unknown_dataset_raises() -> None:
    """An unknown dataset name raises ValueError from build_transform."""
    with pytest.raises(ValueError, match="unknown dataset"):
        build_transform("imagenet")
