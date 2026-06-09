"""Frozen configuration dataclasses (spec §3).

All dataclasses are ``frozen=True`` and have defaults; experiments override fields.
Only ``TempScheduleConfig`` carries validation (strict ``tau>0``, spec §3 [R16]).
``input_dim``/``output_dim`` are dataset-derived in the entrypoint and explicit for
the synthetic path.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelConfig:
    """ReLU MLP architecture (spec §3, §4.3)."""

    input_dim: int
    hidden_dims: tuple[int, ...] = (256, 256)
    output_dim: int = 10


@dataclass(frozen=True)
class OptimConfig:
    """Optimizer configuration (spec §3)."""

    lr: float = 1e-3
    weight_decay: float = 0.0
    name: str = "adam"


@dataclass(frozen=True)
class TempScheduleConfig:
    """Temperature-annealing schedule configuration (spec §3, §4.5)."""

    kind: str = "exponential"
    tau_start: float = 1.0
    tau_end: float = 0.1

    def __post_init__(self) -> None:
        if self.tau_start <= 0.0:
            raise ValueError(f"tau_start must be > 0, got {self.tau_start}")
        if self.tau_end <= 0.0:
            raise ValueError(f"tau_end must be > 0, got {self.tau_end}")


@dataclass(frozen=True)
class RegConfig:
    """Sign-consistency regularizer configuration (spec §3, §4.4 [R1])."""

    lam: float = 0.0
    entropy_eps: float = 1e-6


@dataclass(frozen=True)
class DataConfig:
    """Dataset and dataloader configuration (spec §3 [R8][R25])."""

    dataset: str = "mnist"
    batch_size: int = 128
    eval_batch_size: int = 512
    val_fraction: float = 0.1
    split_seed: int = 0
    drop_last: bool = False
    data_dir: str = "data"
    num_workers: int = 0

    def __post_init__(self) -> None:
        if not 0.0 < self.val_fraction < 1.0:
            raise ValueError(f"val_fraction must be in (0, 1), got {self.val_fraction}")


@dataclass(frozen=True)
class ProbeConfig:
    """Probe neuron/batch configuration, invariant to batch_size sweeps (§3 [R25])."""

    num_neurons: int = 16
    seed: int = 0
    batch_size: int = 512


@dataclass(frozen=True)
class CheckpointConfig:
    """Checkpoint cadence configuration (spec §3)."""

    every_epochs: int = 1
    dir: str = "runs"


@dataclass(frozen=True)
class SurgeryConfig:
    """Surgery k-sweep configuration (spec §3, §4.11)."""

    num_k: int = 21
    random_baseline: bool = True
    tie_break: str = "identity"

    def __post_init__(self) -> None:
        if self.tie_break not in {"zero", "identity"}:
            raise ValueError(
                f"tie_break must be 'zero' or 'identity', got {self.tie_break!r}"
            )


@dataclass(frozen=True)
class WandbConfig:
    """wandb logging configuration (spec §3, §4.6)."""

    project: str = "kill-nonlinearities"
    entity: str | None = None
    mode: str = "online"
    group: str | None = None
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class TrainConfig:
    """Training-loop configuration (spec §3, §4.8)."""

    epochs: int = 20
    seed: int = 0
    device: str = "cpu"
    grad_clip: float | None = None


@dataclass(frozen=True)
class ExperimentConfig:
    """Composes every config section plus a run name (spec §3, §4.13)."""

    name: str = "experiment"
    model: ModelConfig = field(default_factory=lambda: ModelConfig(input_dim=784))
    optim: OptimConfig = field(default_factory=OptimConfig)
    temp_schedule: TempScheduleConfig = field(default_factory=TempScheduleConfig)
    reg: RegConfig = field(default_factory=RegConfig)
    data: DataConfig = field(default_factory=DataConfig)
    probe: ProbeConfig = field(default_factory=ProbeConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    surgery: SurgeryConfig = field(default_factory=SurgeryConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
