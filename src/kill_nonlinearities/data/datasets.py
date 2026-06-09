"""Dataset providers, deterministic dataloaders, and probe selection (spec §4.9)."""

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split
from torchvision import transforms
from torchvision.datasets import CIFAR10, MNIST

from kill_nonlinearities.config import ExperimentConfig

_SYNTHETIC_TRAIN_SIZE = 256
_SYNTHETIC_TEST_SIZE = 64

_NORMALIZE = {
    "mnist": ((0.1307,), (0.3081,)),
    "cifar10": (
        (0.4914, 0.4822, 0.4465),
        (0.2470, 0.2435, 0.2616),
    ),
}


def build_transform(dataset: str) -> transforms.Compose:
    """ToTensor + per-dataset Normalize, with NO train-time augmentation ``[R30]``.

    The same deterministic transform is used for train/val/test so only seeded
    shuffling and initialization remain stochastic.
    """
    if dataset not in _NORMALIZE:
        raise ValueError(f"unknown dataset {dataset!r}")
    mean, std = _NORMALIZE[dataset]
    return transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])


def _make_synthetic_split(
    *, size: int, input_dim: int, output_dim: int, generator: torch.Generator
) -> TensorDataset:
    """A Gaussian-input TensorDataset with uniform random integer labels (no network)."""
    x = torch.randn(size, input_dim, generator=generator)
    y = torch.randint(0, output_dim, (size,), generator=generator)
    return TensorDataset(x, y)


def _build_train_and_test(config: ExperimentConfig) -> tuple[Dataset, Dataset]:
    """Return the canonical (train, test) datasets for the configured provider.

    Only the synthetic provider is exercised by the offline test suite; the
    torchvision providers are added in a later task.
    """
    dataset = config.data.dataset
    if dataset == "synthetic":
        gen = torch.Generator()
        gen.manual_seed(config.data.split_seed)
        train = _make_synthetic_split(
            size=_SYNTHETIC_TRAIN_SIZE,
            input_dim=config.model.input_dim,
            output_dim=config.model.output_dim,
            generator=gen,
        )
        test = _make_synthetic_split(
            size=_SYNTHETIC_TEST_SIZE,
            input_dim=config.model.input_dim,
            output_dim=config.model.output_dim,
            generator=gen,
        )
        return train, test
    if dataset in {"mnist", "cifar10"}:  # pragma: no cover - network download
        transform = build_transform(dataset)
        dataset_cls = MNIST if dataset == "mnist" else CIFAR10
        train = dataset_cls(
            root=config.data.data_dir,
            train=True,
            download=True,
            transform=transform,
        )
        test = dataset_cls(
            root=config.data.data_dir,
            train=False,
            download=True,
            transform=transform,
        )
        return train, test
    raise ValueError(f"unknown dataset {dataset!r}")


def make_dataloaders(
    config: ExperimentConfig,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build (train, val, test) loaders (spec §4.9).

    val is carved from the train split via ``random_split`` with an explicit
    ``torch.Generator(split_seed)``. train shuffles with a seeded generator and
    honors ``drop_last``; val/test use ``shuffle=False, drop_last=False`` and
    ``eval_batch_size`` so downstream analysis concatenates in a stable order.
    """
    train_full, test_dataset = _build_train_and_test(config)

    n_total = len(train_full)  # ty: ignore[invalid-argument-type]
    n_val = round(config.data.val_fraction * n_total)
    n_train = n_total - n_val
    split_gen = torch.Generator()
    split_gen.manual_seed(config.data.split_seed)
    train_dataset, val_dataset = random_split(
        train_full, [n_train, n_val], generator=split_gen
    )

    shuffle_gen = torch.Generator()
    shuffle_gen.manual_seed(config.data.split_seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        drop_last=config.data.drop_last,
        num_workers=config.data.num_workers,
        generator=shuffle_gen,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.data.eval_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=config.data.num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.data.eval_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=config.data.num_workers,
    )
    return train_loader, val_loader, test_loader


def select_probe_neurons(model: object, num: int, seed: int) -> list[tuple[str, int]]:
    """Pick a fixed, seeded set of ``num`` (site, neuron_index) probe pairs (spec §4.9).

    The set is deterministic in ``seed`` and invariant across calls / checkpoints /
    sweep runs ``[R25]`` so a probe GIF reflects weight evolution only. If ``num``
    exceeds the total neuron count, all neurons are returned.
    """
    site_names: tuple[str, ...] = model.site_names  # ty: ignore[unresolved-attribute]
    activations = model.activations  # ty: ignore[unresolved-attribute]
    all_pairs: list[tuple[str, int]] = []
    for name, act in zip(site_names, activations, strict=True):
        width = int(act.mode.shape[0])
        all_pairs.extend((name, i) for i in range(width))

    count = min(num, len(all_pairs))
    gen = torch.Generator()
    gen.manual_seed(seed)
    order = torch.randperm(len(all_pairs), generator=gen).tolist()
    return [all_pairs[i] for i in order[:count]]


def make_probe_batch(loader: DataLoader, size: int, seed: int) -> Tensor:
    """Return a fixed, seeded batch of ``size`` inputs from ``loader``'s dataset (spec §4.9).

    Inputs only (no labels), flattened to ``[size, input_dim]``. Deterministic in
    ``seed`` and identical across calls / checkpoints / sweep runs ``[R25]`` so a
    probe GIF reflects weight evolution only. If ``size`` exceeds the dataset, all
    samples are returned.
    """
    dataset = loader.dataset
    n = len(dataset)  # ty: ignore[invalid-argument-type]
    count = min(size, n)
    gen = torch.Generator()
    gen.manual_seed(seed)
    order = torch.randperm(n, generator=gen)[:count].tolist()
    inputs = [torch.as_tensor(dataset[i][0]).flatten() for i in order]
    return torch.stack(inputs)
