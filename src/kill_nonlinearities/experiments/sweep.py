"""wandb sweep glue: pure config translation + lazy-wandb launch/entry (spec §4.13).

`wandb` is imported LAZILY, only inside `sweep_entry` and `launch_sweep`, so importing this
module (e.g. to use `build_sweep_config` / `config_from_wandb`) never pulls in wandb ([R19]).
"""

from collections.abc import Mapping
from dataclasses import replace

from kill_nonlinearities.config import ExperimentConfig

# The documented BASE default config: every field a sweep does NOT vary is taken from here.
# The swept fields (lr, epochs, batch_size, lam) are overridden per run by `config_from_wandb`.
BASE_CONFIG = ExperimentConfig(name="sweep")

# Flat wandb-config key → the swept value's home in the nested ExperimentConfig (spec §4.13).
_SWEPT_KEYS: frozenset[str] = frozenset({"lr", "epochs", "batch_size", "lam"})


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


def config_from_wandb(wandb_config: Mapping[str, object]) -> ExperimentConfig:
    """Translate a flat wandb config dict into a nested ExperimentConfig ([R9]).

    Flat → nested map: `lr`→`OptimConfig.lr`, `epochs`→`TrainConfig.epochs`,
    `batch_size`→`DataConfig.batch_size`, `lam`→`RegConfig.lam`. All other fields (incl.
    `seed`) come from `BASE_CONFIG`. UNKNOWN keys raise `KeyError`; MISSING swept keys raise
    `KeyError`. Pure: no wandb, no network.
    """
    keys = set(wandb_config)
    unknown = keys - _SWEPT_KEYS
    if unknown:
        raise KeyError(f"unknown wandb config keys: {sorted(unknown)}")
    missing = _SWEPT_KEYS - keys
    if missing:
        raise KeyError(f"missing swept wandb config keys: {sorted(missing)}")

    return replace(
        BASE_CONFIG,
        optim=replace(BASE_CONFIG.optim, lr=wandb_config["lr"]),
        train=replace(BASE_CONFIG.train, epochs=wandb_config["epochs"]),
        data=replace(BASE_CONFIG.data, batch_size=wandb_config["batch_size"]),
        reg=replace(BASE_CONFIG.reg, lam=wandb_config["lam"]),
    )
