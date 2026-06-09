"""Functional test for analysis.statistics.collect_history (spec §4.10, [R11])."""

from unittest import mock

import torch
from torch.utils.data import DataLoader, TensorDataset

from kill_nonlinearities.analysis.statistics import FrameStats, collect_history
from kill_nonlinearities.config import ExperimentConfig, ModelConfig
from kill_nonlinearities.models.mlp import ReLUMLP
from kill_nonlinearities.training.checkpoint import save_checkpoint


def _val_loader() -> DataLoader:
    x = torch.randn(6, 4)
    ds = TensorDataset(x, torch.zeros(6, dtype=torch.int64))
    return DataLoader(ds, batch_size=3, shuffle=False)


def test_collect_history_one_frame_per_checkpoint(tmp_path) -> None:
    torch.manual_seed(0)
    model_config = ModelConfig(input_dim=4, hidden_dims=(3, 2), output_dim=2)
    config = ExperimentConfig(name="hist", model=model_config)

    def model_factory() -> ReLUMLP:
        return ReLUMLP(model_config)

    # Two distinct checkpoints (mutate weights between saves).
    model = model_factory()
    paths = []
    for step in (0, 5):
        with torch.no_grad():
            for p in model.parameters():
                p.add_(0.1)
        paths.append(save_checkpoint(model, step, config, tmp_path))

    probe_batch = torch.randn(2, 4)
    probe_neurons = [("relu0", 0), ("relu1", 1)]

    frames = collect_history(
        paths,
        model_factory,
        probe_batch,
        _val_loader(),
        probe_neurons,
        device="cpu",
    )

    assert len(frames) == 2
    assert all(isinstance(f, FrameStats) for f in frames)
    assert [f.step for f in frames] == [0, 5]

    f0 = frames[0]
    # Probe activations: one vector per probe neuron, length == probe batch size.
    assert set(f0.probe_activations) == {("relu0", 0), ("relu1", 1)}
    assert f0.probe_activations[("relu0", 0)].shape == (2,)
    # q_by_site: one entry per site, width == that layer's hidden dim.
    assert set(f0.q_by_site) == {"relu0", "relu1"}
    assert f0.q_by_site["relu0"].shape == (3,)
    assert f0.q_by_site["relu1"].shape == (2,)


def test_collect_history_moves_rebuilt_model_to_device(tmp_path) -> None:
    """collect_history owns the .to(device) move on every factory-built model.

    A factory that returns a CPU model would crash on a CUDA run if the model
    were used unmoved; this pins that collect_history calls .to(device) on each
    rebuilt model (capstone fix). We record the device argument each .to receives.
    """
    torch.manual_seed(0)
    model_config = ModelConfig(input_dim=4, hidden_dims=(3,), output_dim=2)
    config = ExperimentConfig(name="hist-dev", model=model_config)

    model = ReLUMLP(model_config)
    path = save_checkpoint(model, 0, config, tmp_path)

    moved_to: list[object] = []
    real_to = ReLUMLP.to

    def recording_to(self: ReLUMLP, device: object) -> ReLUMLP:
        moved_to.append(device)
        return real_to(self, device)  # ty: ignore[no-matching-overload]

    with mock.patch.object(ReLUMLP, "to", recording_to):
        frames = collect_history(
            [path],
            lambda: ReLUMLP(model_config),
            torch.randn(2, 4),
            _val_loader(),
            [("relu0", 0)],
            device="cpu",
        )

    assert len(frames) == 1
    # The rebuilt model was explicitly moved to the requested device.
    assert "cpu" in moved_to
