"""Launch a wandb hyperparameter sweep over {lr, epochs, batch_size, lambda}.

Requires a wandb account — run ``uv run wandb login`` first. The sweep agent runs
``run_experiment`` once per trial, logs metrics/artifacts to wandb, and optimizes the
``val/acc`` metric. Each trial gets a unique run name so trials don't share an output dir.

Usage:
    uv run wandb login
    uv run --no-sync python scripts/launch_sweep.py --method grid --count 24

Edit ``DEFAULT_GRID`` below (or the CLI) to change the search space.
"""

from __future__ import annotations

import argparse

from kill_nonlinearities.experiments.sweep import launch_sweep

DEFAULT_GRID: dict[str, list[object]] = {
    "lr": [1e-3, 3e-3],
    "epochs": [8, 16],
    "batch_size": [64, 128],
    "lam": [0.0, 0.01, 0.05, 0.1],
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch a wandb sweep over {lr, epochs, batch_size, lambda}.",
    )
    parser.add_argument(
        "--method",
        default="grid",
        choices=["grid", "random", "bayes"],
        help="wandb sweep search strategy",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=24,
        help="number of trials the agent runs",
    )
    args = parser.parse_args()

    sweep_id = launch_sweep(DEFAULT_GRID, method=args.method, count=args.count)
    print(f"sweep launched: {sweep_id}")


if __name__ == "__main__":
    main()
