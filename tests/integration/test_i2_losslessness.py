"""Integration test for invariant I2 — selection-set losslessness (spec §6 [R6]).

Build an UNTRAINED model with direct, large +/- biases so that, on the selection
set, at least one neuron has q==0 (ZERO-eligible) and at least one has q==1
(IDENTITY-eligible), each with true H(q)==0. Filter to exactly {H(q)==0} (NOT
select_topk), convert only those, and assert the logits are torch.equal-unchanged
on that same selection set.
"""

import torch
from torch.utils.data import DataLoader, TensorDataset

from kill_nonlinearities.analysis.selection import assign_modes
from kill_nonlinearities.analysis.statistics import (
    collect_pre_activations,
    neuron_stats,
)
from kill_nonlinearities.config import ModelConfig
from kill_nonlinearities.models.mlp import ReLUMLP
from kill_nonlinearities.surgery.apply import apply_modes


def test_i2_selection_set_losslessness() -> None:
    torch.manual_seed(0)
    input_dim = 4
    model = ReLUMLP(ModelConfig(input_dim=input_dim, hidden_dims=(4,), output_dim=3))

    # Force entropy-0 neurons via direct biases on the single hidden layer:
    #   neuron 0: huge NEGATIVE bias -> always z<0 -> q==0 (ZERO, H=0).
    #   neuron 1: huge POSITIVE bias -> always z>0 -> q==1 (IDENTITY, H=0).
    #   neurons 2,3: zero weight+bias mixed by inputs -> interior q (H>0), excluded.
    with torch.no_grad():
        linear = model.linears[0]
        linear.weight.zero_()
        linear.bias.zero_()
        linear.bias[0] = -1.0e3  # always negative -> q==0
        linear.bias[1] = 1.0e3  # always positive -> q==1
        # neurons 2,3: let the inputs decide -> mixed signs across the batch.
        linear.weight[2, 0] = 1.0
        linear.weight[3, 1] = 1.0

    # Selection-set loader: a small fixed batch of inputs with mixed signs.
    xs = torch.tensor(
        [
            [1.0, -1.0, 0.5, -0.5],
            [-1.0, 1.0, -0.5, 0.5],
            [2.0, -2.0, 1.0, -1.0],
            [-2.0, 2.0, -1.0, 1.0],
        ]
    )
    loader = DataLoader(
        TensorDataset(xs, torch.zeros(xs.shape[0], dtype=torch.int64)),
        batch_size=2,
        shuffle=False,
    )

    # Baseline logits on the selection set (k=0, all RELU).
    model.eval()
    with torch.no_grad():
        logits_before = model(xs).logits.clone()

    # Stats on the SELECTION SET; filter to exactly {H(q)==0}.
    pre = collect_pre_activations(model, loader, device="cpu")
    stats = neuron_stats(pre)
    entropy_zero = [s for s in stats if s.entropy == 0.0]

    # Non-vacuous: at least one ZERO (q==0) and one IDENTITY (q==1) neuron.
    has_zero = any(s.q == 0.0 for s in entropy_zero)
    has_identity = any(s.q == 1.0 for s in entropy_zero)
    assert has_zero, "expected >=1 q==0 (ZERO) entropy-0 neuron"
    assert has_identity, "expected >=1 q==1 (IDENTITY) entropy-0 neuron"

    # Convert ONLY the entropy-0 neurons (NOT via select_topk), full-width modes.
    widths = {site: int(p.shape[1]) for site, p in pre.items()}
    modes = assign_modes(entropy_zero, tie_break="identity", widths=widths)
    apply_modes(model, modes)

    # Logits on the SAME selection set are bit-for-bit unchanged.
    with torch.no_grad():
        logits_after = model(xs).logits
    assert torch.equal(logits_after, logits_before)


def test_i2_cross_layer_identity_losslessness_two_sites() -> None:
    """I2 across TWO hidden layers: a site-0 IDENTITY passthrough flows downstream.

    On a (4, 4) model, site 0 carries an entropy-0 ZERO (q==0) and an entropy-0
    IDENTITY (q==1) neuron via direct large biases; site 1 also carries an
    entropy-0 ZERO and IDENTITY neuron. Converting ONLY the entropy-0 neurons
    (so site 0's IDENTITY passthrough feeds the next Linear + SelectiveReLU) must
    leave logits torch.equal-unchanged on the selection set.
    """
    torch.manual_seed(0)
    input_dim = 4
    model = ReLUMLP(ModelConfig(input_dim=input_dim, hidden_dims=(4, 4), output_dim=3))

    big = 1.0e3
    with torch.no_grad():
        # --- site 0 (relu0): force two entropy-0 neurons, two interior neurons. ---
        lin0 = model.linears[0]
        lin0.weight.zero_()
        lin0.bias.zero_()
        lin0.bias[0] = -big  # always z<0 -> q==0 (ZERO, H=0)
        lin0.bias[1] = +big  # always z>0 -> q==1 (IDENTITY, H=0)
        lin0.weight[2, 0] = 1.0  # interior: sign follows input feature 0
        lin0.weight[3, 1] = 1.0  # interior: sign follows input feature 1

        # --- site 1 (relu1): force two entropy-0 neurons via large biases. ---
        # site-0 IDENTITY (neuron 1) passes its huge +big value into lin1; keep
        # lin1's weights on the interior site-0 neurons so the biases dominate the
        # entropy-0 site-1 neurons' sign deterministically.
        lin1 = model.linears[1]
        lin1.weight.zero_()
        lin1.bias.zero_()
        lin1.bias[0] = -big  # always z<0 -> q==0 (ZERO, H=0)
        lin1.bias[1] = +big  # always z>0 -> q==1 (IDENTITY, H=0)
        lin1.weight[2, 2] = 1.0  # interior: follows site-0 interior neuron 2
        lin1.weight[3, 3] = 1.0  # interior: follows site-0 interior neuron 3

    # Selection-set loader: a small fixed batch of inputs with mixed signs.
    xs = torch.tensor(
        [
            [1.0, -1.0, 0.5, -0.5],
            [-1.0, 1.0, -0.5, 0.5],
            [2.0, -2.0, 1.0, -1.0],
            [-2.0, 2.0, -1.0, 1.0],
        ]
    )
    loader = DataLoader(
        TensorDataset(xs, torch.zeros(xs.shape[0], dtype=torch.int64)),
        batch_size=2,
        shuffle=False,
    )

    model.eval()
    with torch.no_grad():
        logits_before = model(xs).logits.clone()

    pre = collect_pre_activations(model, loader, device="cpu")
    stats = neuron_stats(pre)
    entropy_zero = [s for s in stats if s.entropy == 0.0]

    # Non-vacuous: site 0 has BOTH a q==0 (ZERO) and a q==1 (IDENTITY) entropy-0
    # neuron, and at least one IDENTITY entropy-0 neuron exists at site 0.
    site0_zero = [s for s in entropy_zero if s.site == "relu0" and s.q == 0.0]
    site0_identity = [s for s in entropy_zero if s.site == "relu0" and s.q == 1.0]
    assert site0_zero, "expected >=1 site-0 q==0 (ZERO) entropy-0 neuron"
    assert site0_identity, "expected >=1 site-0 q==1 (IDENTITY) entropy-0 neuron"
    # The cross-layer property under test: a site-0 IDENTITY exists and feeds
    # downstream layers once converted.
    assert any(s.site == "relu1" for s in entropy_zero)

    widths = {site: int(p.shape[1]) for site, p in pre.items()}
    modes = assign_modes(entropy_zero, tie_break="identity", widths=widths)
    apply_modes(model, modes)

    with torch.no_grad():
        logits_after = model(xs).logits
    assert torch.equal(logits_after, logits_before)
