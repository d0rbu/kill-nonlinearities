"""Frozen configuration dataclasses (spec §3).

All dataclasses are ``frozen=True`` and have defaults; experiments override fields.
Only ``TempScheduleConfig`` carries validation (strict ``tau>0``, spec §3 [R16]).
``input_dim``/``output_dim`` are dataset-derived in the entrypoint and explicit for
the synthetic path.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelConfig:
    """Model architecture: ReLU MLP or ReLU CNN (spec §3, §4.3; conv spec 2026-06-10).

    ``kind="mlp"`` (default) uses ``input_dim``/``hidden_dims`` and ignores the
    conv fields; ``kind="cnn"`` uses ``in_channels``/``image_size``/
    ``conv_channels`` for the conv blocks and ``hidden_dims`` for the FC head
    (``input_dim`` is ignored — the head's fan-in is derived).
    """

    input_dim: int
    hidden_dims: tuple[int, ...] = (256, 256)
    output_dim: int = 10
    kind: str = "mlp"
    in_channels: int = 3
    image_size: int = 32
    conv_channels: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.kind not in {"mlp", "cnn"}:
            raise ValueError(f"kind must be 'mlp' or 'cnn', got {self.kind!r}")
        if self.kind == "cnn":
            if not self.conv_channels:
                raise ValueError("cnn models require a non-empty conv_channels")
            down = 2 ** len(self.conv_channels)
            if self.image_size % down != 0:
                raise ValueError(
                    f"image_size {self.image_size} must be divisible by {down} "
                    f"(one 2x2 max-pool per conv block)"
                )


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
    """Sign-consistency regularizer configuration (spec §3, §4.4 [R1]).

    ``granularity`` selects the loss's pooling unit (conv spec 2026-06-10
    addendum): ``"neuron"`` (default) takes the entropy of each scalar unit's
    batch-mean ``p_i``; ``"channel"`` pools each conv channel's positions into
    the sample dimension before the entropy (groups of size 1 elsewhere, so it
    equals ``"neuron"`` for MLP/fc sites). Analysis and surgery stay per-neuron
    either way — the knob changes only the training incentive.
    """

    lam: float = 0.0
    entropy_eps: float = 1e-6
    granularity: str = "neuron"

    def __post_init__(self) -> None:
        if self.granularity not in {"neuron", "channel"}:
            raise ValueError(
                f"granularity must be 'neuron' or 'channel', got {self.granularity!r}"
            )


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
        if self.num_k < 2:
            # make_k_grid divides by (num_k - 1); fewer than 2 points is undefined.
            raise ValueError(f"num_k must be >= 2, got {self.num_k}")


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
