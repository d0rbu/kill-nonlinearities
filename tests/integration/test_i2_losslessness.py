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
