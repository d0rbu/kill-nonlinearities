"""wandb sweep glue: pure config translation + lazy-wandb launch/entry (spec §4.13).

`wandb` is imported LAZILY, only inside `sweep_entry` and `launch_sweep`, so importing this
module (e.g. to use `build_sweep_config` / `config_from_wandb`) never pulls in wandb ([R19]).
"""

from collections.abc import Mapping


def build_sweep_config(
    grids: Mapping[str, list[object]], method: str
) -> dict[str, object]:
    """Build a wandb sweep config from per-parameter value grids ([R20]).

    The `metric` name MUST match a trainer/run-logged scalar key (`val/acc`); `goal` is
    fixed to `maximize`. `grids` keys are the swept fields (`lr`, `epochs`, `batch_size`,
    `lam`); each maps to its list of candidate values.
    """
    return {
        "method": method,
        "metric": {"name": "val/acc", "goal": "maximize"},
        "parameters": {key: {"values": values} for key, values in grids.items()},
    }
