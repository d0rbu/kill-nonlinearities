"""run_experiment: wire the full Phase-1a data flow into one ExperimentResult.

Implements spec §4.13 and the §5 data flow. wandb is imported LAZILY only inside
WandbLogger (constructed here solely when no logger is supplied); the pure pipeline
never imports wandb.
"""

from dataclasses import dataclass
from pathlib import Path

from kill_nonlinearities.analysis.statistics import FrameStats, NeuronStats
from kill_nonlinearities.models.mlp import ReLUMLP
from kill_nonlinearities.surgery.apply import KPoint
from kill_nonlinearities.training.trainer import TrainResult


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
