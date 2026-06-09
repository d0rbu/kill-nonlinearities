"""Unit tests for probe selection determinism (spec §4.9, §7 probe fixity)."""


from kill_nonlinearities.config import ModelConfig
from kill_nonlinearities.data.datasets import select_probe_neurons
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
