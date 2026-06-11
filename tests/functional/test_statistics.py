"""Functional tests for analysis.statistics (spec §4.10, §7).

Stats are verified against a hand-computed expectation on a tiny loader, using
the exact ``model(x).pre_activations`` path that surgery eval also uses (I2).
"""

import math

import torch
from torch.utils.data import DataLoader, TensorDataset

from kill_nonlinearities.analysis.statistics import (
    NeuronStats,
    collect_pre_activations,
    neuron_stats,
)
from kill_nonlinearities.config import ModelConfig
from kill_nonlinearities.models.mlp import ReLUMLP


def _two_sample_loader(x: torch.Tensor) -> DataLoader:
    ds = TensorDataset(x, torch.zeros(x.shape[0], dtype=torch.int64))
    return DataLoader(ds, batch_size=1, shuffle=False)


def test_collect_pre_activations_concatenates_over_batches() -> None:
    torch.manual_seed(0)
    model = ReLUMLP(ModelConfig(input_dim=4, hidden_dims=(3,), output_dim=2))
    x = torch.randn(5, 4)
    loader = _two_sample_loader(x)

    collected = collect_pre_activations(model, loader, device="cpu")

    assert set(collected) == set(model.site_names)
    # M=5 samples, N=3 hidden units at the single site.
    site = model.site_names[0]
    assert collected[site].shape == (5, 3)

    # Bit-exact against forwards with the loader's own batching (the same
    # model(x) path on the same shapes -> the same kernels, so exact equality
    # is guaranteed; this is the invariant analysis/surgery rely on, I2).
    model.eval()
    with torch.no_grad():
        ref_batched = torch.cat([model(xb).pre_activations[0] for xb, _ in loader])
    torch.testing.assert_close(collected[site], ref_batched, rtol=0, atol=0)

    # A single full-batch forward agrees only up to float reassociation: the
    # batch size selects the GEMM kernel/reduction order, which torch does not
    # keep bitwise-stable (1-ULP drift observed on the macOS arm64 cp313 wheel).
    with torch.no_grad():
        ref_full = model(x).pre_activations[0]
    torch.testing.assert_close(collected[site], ref_full)


def test_neuron_stats_vs_hand_computed() -> None:
    # One site, two neurons. Column 0: pre-acts (1, -2, 3) -> q=2/3.
    #                         Column 1: pre-acts (-1,-1,-1) -> q=0 (dead).
    pre = torch.tensor([[1.0, -1.0], [-2.0, -1.0], [3.0, -1.0]])
    by_site = {"relu0": pre}

    stats = neuron_stats(by_site)

    assert [type(s) for s in stats] == [NeuronStats, NeuronStats]
    s0, s1 = stats
    assert s0.site == "relu0" and s0.index == 0
    assert s1.site == "relu0" and s1.index == 1

    # q via strict z>0: column 0 -> 2/3, column 1 -> 0.
    assert math.isclose(s0.q, 2.0 / 3.0, rel_tol=0, abs_tol=1e-7)
    assert s1.q == 0.0

    # entropy in nats: H(2/3) = -(2/3 ln 2/3 + 1/3 ln 1/3); H(0) = 0 exact.
    q0 = 2.0 / 3.0
    expected_h0 = -(q0 * math.log(q0) + (1 - q0) * math.log(1 - q0))
    assert math.isclose(s0.entropy, expected_h0, rel_tol=0, abs_tol=1e-6)
    assert s1.entropy == 0.0

    # mean pre-activation per neuron.
    assert math.isclose(s0.mean_pre, (1.0 - 2.0 + 3.0) / 3.0, rel_tol=0, abs_tol=1e-6)
    assert math.isclose(s1.mean_pre, -1.0, rel_tol=0, abs_tol=1e-6)
