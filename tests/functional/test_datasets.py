"""Functional tests for data/datasets.py (spec §4.9, §7).

Tests use ONLY the synthetic provider — no network, no torchvision download.
Determinism follows the spec §8 recipe (seeded generators, num_workers=0).
"""

import torch
from torch import Tensor

from kill_nonlinearities.config import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    ProbeConfig,
)
from kill_nonlinearities.data.datasets import make_dataloaders


def _synthetic_config(
    *,
    batch_size: int = 8,
    eval_batch_size: int = 16,
    val_fraction: float = 0.2,
    split_seed: int = 0,
    drop_last: bool = False,
    seed: int = 0,
) -> ExperimentConfig:
    """A tiny, fully-synthetic ExperimentConfig (no network, no download)."""
    return ExperimentConfig(
        name="synthetic-test",
        model=ModelConfig(input_dim=12, hidden_dims=(8,), output_dim=3),
        data=DataConfig(
            dataset="synthetic",
            batch_size=batch_size,
            eval_batch_size=eval_batch_size,
            val_fraction=val_fraction,
            split_seed=split_seed,
            drop_last=drop_last,
            num_workers=0,
        ),
        probe=ProbeConfig(num_neurons=4, seed=0, batch_size=10),
    )
    # `seed` lives on TrainConfig; left at its default here (unused by loaders).


def test_synthetic_make_dataloaders_returns_three_loaders() -> None:
    """make_dataloaders returns exactly (train, val, test)."""
    config = _synthetic_config()
    loaders = make_dataloaders(config)
    assert len(loaders) == 3
    train, val, test = loaders
    assert train is not val
    assert val is not test


def test_synthetic_train_batch_shapes() -> None:
    """Train batches are (x, y) with x flattened to [B, input_dim] and int64 y."""
    config = _synthetic_config(batch_size=8)
    train, _, _ = make_dataloaders(config)
    x, y = next(iter(train))
    assert isinstance(x, Tensor)
    assert isinstance(y, Tensor)
    assert x.shape[0] <= 8
    assert x.shape[1] == config.model.input_dim
    assert x.dtype == torch.float32
    assert y.dtype == torch.int64
    assert int(y.min()) >= 0
    assert int(y.max()) < config.model.output_dim


def test_synthetic_eval_loaders_use_eval_batch_size() -> None:
    """val/test loaders batch with eval_batch_size, not the train batch_size."""
    config = _synthetic_config(batch_size=8, eval_batch_size=16)
    _, val, test = make_dataloaders(config)
    assert val.batch_size == 16
    assert test.batch_size == 16


def test_synthetic_val_carved_from_train_by_fraction() -> None:
    """val is carved from train; train+val cover the full train split."""
    config = _synthetic_config(val_fraction=0.2)
    train, val, _ = make_dataloaders(config)
    n_train = len(train.dataset)  # ty: ignore[invalid-argument-type]
    n_val = len(val.dataset)  # ty: ignore[invalid-argument-type]
    total = n_train + n_val
    assert n_val == round(0.2 * total)
    assert n_train == total - n_val
    assert n_val > 0
    assert n_train > 0


def _stack_dataset(dataset: object) -> tuple[Tensor, Tensor]:
    """Materialize a (small) dataset's inputs and labels into stacked tensors."""
    xs: list[Tensor] = []
    ys: list[Tensor] = []
    for x, y in dataset:  # ty: ignore[not-iterable]
        xs.append(x)
        ys.append(torch.as_tensor(y))
    return torch.stack(xs), torch.stack(ys)


def test_synthetic_split_is_deterministic_under_split_seed() -> None:
    """Same config → identical val/train partition (spec §8 determinism recipe)."""
    config = _synthetic_config(split_seed=7)
    train_a, val_a, _ = make_dataloaders(config)
    train_b, val_b, _ = make_dataloaders(config)

    xa, ya = _stack_dataset(val_a.dataset)
    xb, yb = _stack_dataset(val_b.dataset)
    assert torch.equal(xa, xb)
    assert torch.equal(ya, yb)

    txa, tya = _stack_dataset(train_a.dataset)
    txb, tyb = _stack_dataset(train_b.dataset)
    assert torch.equal(txa, txb)
    assert torch.equal(tya, tyb)


def test_synthetic_split_changes_with_split_seed() -> None:
    """A different split_seed yields a different (non-degenerate) val partition."""
    val_seed0 = _stack_dataset(
        make_dataloaders(_synthetic_config(split_seed=0))[1].dataset
    )
    val_seed1 = _stack_dataset(
        make_dataloaders(_synthetic_config(split_seed=1))[1].dataset
    )
    assert not torch.equal(val_seed0[0], val_seed1[0])


def _concat_loader_inputs(loader: object) -> Tensor:
    """Concatenate all batch inputs from a loader in iteration order."""
    return torch.cat([x for x, _ in loader])  # ty: ignore[not-iterable]


def test_val_iteration_order_is_stable() -> None:
    """val iterates in a fixed order across repeated passes (shuffle=False)."""
    _, val, _ = make_dataloaders(_synthetic_config())
    first = _concat_loader_inputs(val)
    second = _concat_loader_inputs(val)
    assert torch.equal(first, second)


def test_test_iteration_order_is_stable() -> None:
    """test iterates in a fixed order across repeated passes (shuffle=False)."""
    _, _, test = make_dataloaders(_synthetic_config())
    first = _concat_loader_inputs(test)
    second = _concat_loader_inputs(test)
    assert torch.equal(first, second)


def test_test_order_stable_across_make_dataloaders_calls() -> None:
    """test order is identical across independent make_dataloaders calls."""
    _, _, test_a = make_dataloaders(_synthetic_config())
    _, _, test_b = make_dataloaders(_synthetic_config())
    assert torch.equal(_concat_loader_inputs(test_a), _concat_loader_inputs(test_b))


def test_val_keeps_all_samples_drop_last_false() -> None:
    """val never drops a partial final batch (drop_last=False)."""
    config = _synthetic_config(val_fraction=0.2, eval_batch_size=7)
    _, val, _ = make_dataloaders(config)
    n_seen = sum(x.shape[0] for x, _ in val)
    assert n_seen == len(val.dataset)  # ty: ignore[invalid-argument-type]
