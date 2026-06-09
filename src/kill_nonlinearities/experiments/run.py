"""run_experiment: wire the full Phase-1a data flow into one ExperimentResult.

Implements spec §4.13 and the §5 data flow. wandb is imported LAZILY only inside
WandbLogger (constructed here solely when no logger is supplied); the pure pipeline
never imports wandb.
"""

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from kill_nonlinearities.analysis.selection import (
    make_k_grid,
    rank_by_entropy,
    rank_random,
)
from kill_nonlinearities.analysis.statistics import (
    FrameStats,
    NeuronStats,
    collect_history,
    collect_pre_activations,
    neuron_stats,
)
from kill_nonlinearities.config import (
    CheckpointConfig,
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    OptimConfig,
    ProbeConfig,
    RegConfig,
    SurgeryConfig,
    TempScheduleConfig,
    TrainConfig,
    WandbConfig,
)
from kill_nonlinearities.data.datasets import (
    make_dataloaders,
    make_probe_batch,
    select_probe_neurons,
)
from kill_nonlinearities.models.mlp import ReLUMLP
from kill_nonlinearities.regularization.surrogate import soft_sign
from kill_nonlinearities.surgery.apply import KPoint, evaluate_accuracy, k_sweep
from kill_nonlinearities.training.logging import Logger, NullLogger
from kill_nonlinearities.training.trainer import TrainResult, train
from kill_nonlinearities.viz.animation import (
    render_activation_gif,
    render_qi_bimodality_gif,
)
from kill_nonlinearities.viz.plots import (
    plot_acc_vs_k,
    plot_entropy_map,
    plot_loss_curves,
    plot_mean_pre_dist,
    plot_per_layer_entropy,
    plot_soft_vs_hard,
)


@dataclass
class ExperimentResult:
    """Aggregated outputs of a single experiment run (spec §4.13)."""

    model: ReLUMLP
    train_result: TrainResult
    neuron_stats: list[NeuronStats]
    k_points: list[KPoint]
    random_k_points: list[KPoint]
    frames: list[FrameStats]
    artifact_paths: dict[str, Path]


def _seed_everything(seed: int) -> None:
    """Re-seed every RNG from ``seed`` (spec §5 step 1, §8 recipe)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_experiment(
    config: ExperimentConfig,
    logger: Logger | None = None,
) -> ExperimentResult:
    """Run the full train -> analyze -> surgery -> visualize pipeline (spec §5)."""
    # 1. Seed (all three RNGs, first thing) + device + logger.
    _seed_everything(config.train.seed)
    device = config.train.device
    if logger is None:  # pragma: no cover - lazy wandb glue, smoke-only per §7 [R23]
        from kill_nonlinearities.training.logging import WandbLogger

        logger = WandbLogger(config.wandb)
    logger.log_config({"name": config.name})

    run_dir = Path(config.checkpoint.dir) / config.name
    run_dir.mkdir(parents=True, exist_ok=True)

    # 2. Data + model + probe (probe neurons/batch are fixed across checkpoints).
    train_loader, val_loader, test_loader = make_dataloaders(config)
    model = ReLUMLP(config.model).to(device)
    probe_neurons = select_probe_neurons(
        model, config.probe.num_neurons, config.probe.seed
    )
    probe_batch = make_probe_batch(
        val_loader, config.probe.batch_size, config.probe.seed
    )

    # 3. Train (history + checkpoints; scalars streamed live).
    train_result = train(model, train_loader, val_loader, config, logger)

    # 4. Analyze on the val (selection) split.
    pre_by_site = collect_pre_activations(model, val_loader, device)
    stats = neuron_stats(pre_by_site)
    total = len(stats)

    # 5. Rank + k-grid + sweep on val AND test. Random baseline uses the run seed.
    ranked = rank_by_entropy(stats)
    ranked_random = rank_random(stats, config.train.seed)
    k_grid = make_k_grid(total, config.surgery.num_k)
    k_points = k_sweep(model, ranked, k_grid, val_loader, test_loader, device)
    random_k_points = (
        k_sweep(model, ranked_random, k_grid, val_loader, test_loader, device)
        if config.surgery.random_baseline
        else []
    )

    # 6. Soft p_i (final tau, on the fixed probe batch) vs hard q_i.
    final_tau = (
        config.temp_schedule.tau_start
        if config.temp_schedule.kind == "constant"
        else config.temp_schedule.tau_end
    )
    model.eval()
    with torch.no_grad():
        out = model(probe_batch.to(device))
        soft_p = torch.cat(
            [soft_sign(z, final_tau).mean(0) for z in out.pre_activations]
        )
    hard_q = torch.tensor([s.q for s in stats], dtype=soft_p.dtype)

    # 7. Per-checkpoint frames for both GIFs (one FrameStats per checkpoint).
    frames = collect_history(
        train_result.checkpoint_paths,
        lambda: ReLUMLP(config.model),
        probe_batch,
        val_loader,
        probe_neurons,
        device,
    )

    # 8. Render + log all artifacts (each renderer takes a FULL FILE path).
    lossless_prefix = sum(1 for s in stats if s.q in (0.0, 1.0))
    artifact_paths: dict[str, Path] = {
        "loss_curves": plot_loss_curves(
            train_result.history, run_dir / "loss_curves.png"
        ),
        "mean_pre_dist": plot_mean_pre_dist(stats, run_dir / "mean_pre_dist.png"),
        "entropy_map": plot_entropy_map(stats, run_dir / "entropy_map.png"),
        "per_layer_entropy": plot_per_layer_entropy(
            stats, run_dir / "per_layer_entropy.png"
        ),
        "acc_vs_k": plot_acc_vs_k(
            k_points,
            random_k_points,
            total,
            lossless_prefix,
            run_dir / "acc_vs_k.png",
        ),
        "soft_vs_hard": plot_soft_vs_hard(soft_p, hard_q, run_dir / "soft_vs_hard.png"),
        "activation_gif": render_activation_gif(frames, run_dir / "activation.gif"),
        "qi_bimodality_gif": render_qi_bimodality_gif(
            frames, run_dir / "qi_bimodality.gif"
        ),
    }
    for key, path in artifact_paths.items():
        if key.endswith("gif"):
            logger.log_video(key, path)
        else:
            logger.log_image(key, path)

    # Log the sweep metric under the exact key the sweep config maximizes (§4.13).
    final_val_acc = evaluate_accuracy(model, val_loader, device)
    final_step = train_result.history[-1].step if train_result.history else 0
    logger.log_scalars({"val/acc": float(final_val_acc)}, step=final_step)
    logger.finish()

    return ExperimentResult(
        model=model,
        train_result=train_result,
        neuron_stats=stats,
        k_points=k_points,
        random_k_points=random_k_points,
        frames=frames,
        artifact_paths=artifact_paths,
    )


def config_from_json(path: Path) -> ExperimentConfig:
    """Load a nested ExperimentConfig from a JSON file (spec §4.13 CLI).

    Top-level keys map to sub-config sections; each section is a flat object of
    that dataclass's fields. Missing sections fall back to their defaults. Tuple
    fields (``hidden_dims``, ``tags``) accept JSON arrays and are coerced to tuples.
    """
    raw = json.loads(Path(path).read_text())

    def _model(d: dict[str, object]) -> ModelConfig:
        kwargs = dict(d)
        if "hidden_dims" in kwargs:
            dims: object = kwargs["hidden_dims"]
            kwargs["hidden_dims"] = tuple(dims)  # ty: ignore[invalid-argument-type]
        return ModelConfig(**kwargs)  # ty: ignore[invalid-argument-type]

    def _wandb(d: dict[str, object]) -> WandbConfig:
        kwargs = dict(d)
        if "tags" in kwargs:
            tags: object = kwargs["tags"]
            kwargs["tags"] = tuple(tags)  # ty: ignore[invalid-argument-type]
        return WandbConfig(**kwargs)  # ty: ignore[invalid-argument-type]

    base = ExperimentConfig()
    return ExperimentConfig(
        name=str(raw.get("name", base.name)),
        model=_model(raw["model"]) if "model" in raw else base.model,  # type: ignore[arg-type]
        optim=OptimConfig(**raw["optim"]) if "optim" in raw else base.optim,  # type: ignore[arg-type]
        temp_schedule=(
            TempScheduleConfig(**raw["temp_schedule"])  # type: ignore[arg-type]
            if "temp_schedule" in raw
            else base.temp_schedule
        ),
        reg=RegConfig(**raw["reg"]) if "reg" in raw else base.reg,  # type: ignore[arg-type]
        data=DataConfig(**raw["data"]) if "data" in raw else base.data,  # type: ignore[arg-type]
        probe=ProbeConfig(**raw["probe"]) if "probe" in raw else base.probe,  # type: ignore[arg-type]
        checkpoint=(
            CheckpointConfig(**raw["checkpoint"])  # type: ignore[arg-type]
            if "checkpoint" in raw
            else base.checkpoint
        ),
        surgery=(
            SurgeryConfig(**raw["surgery"])  # type: ignore[arg-type]
            if "surgery" in raw
            else base.surgery
        ),
        wandb=_wandb(raw["wandb"]) if "wandb" in raw else base.wandb,  # type: ignore[arg-type]
        train=TrainConfig(**raw["train"]) if "train" in raw else base.train,  # type: ignore[arg-type]
    )


def main(argv: list[str] | None = None) -> None:  # pragma: no cover - CLI glue
    parser = argparse.ArgumentParser(description="Run a Phase-1a experiment.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a JSON ExperimentConfig; default uses all-defaults config.",
    )
    args = parser.parse_args(argv)
    config = (
        config_from_json(Path(args.config))
        if args.config is not None
        else ExperimentConfig()
    )
    run_experiment(config, logger=NullLogger())


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
