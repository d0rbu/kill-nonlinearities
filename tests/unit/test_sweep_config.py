"""Unit tests for the PURE sweep helpers (spec §4.13, §7).

`build_sweep_config` and `config_from_wandb` are pure: no wandb, no network, no torch RNG.
"""


from kill_nonlinearities.experiments.sweep import build_sweep_config


def test_build_sweep_config_exact_dict() -> None:
    """grids + method → the exact wandb sweep dict, incl. the frozen metric block ([R20])."""
    grids: dict[str, list[object]] = {
        "lr": [1e-3, 1e-2],
        "epochs": [10, 20],
        "batch_size": [64, 128],
        "lam": [0.0, 0.1],
    }

    result = build_sweep_config(grids, "grid")

    assert result == {
        "method": "grid",
        "metric": {"name": "val/acc", "goal": "maximize"},
        "parameters": {
            "lr": {"values": [1e-3, 1e-2]},
            "epochs": {"values": [10, 20]},
            "batch_size": {"values": [64, 128]},
            "lam": {"values": [0.0, 0.1]},
        },
    }


def test_build_sweep_config_passes_method_through() -> None:
    """The `method` argument is copied verbatim into the config."""
    result = build_sweep_config({"lr": [1e-3]}, "bayes")

    assert result["method"] == "bayes"
    assert result["metric"] == {"name": "val/acc", "goal": "maximize"}
    assert result["parameters"] == {"lr": {"values": [1e-3]}}
