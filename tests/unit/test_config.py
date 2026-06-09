"""Unit tests for the frozen config dataclasses (spec §3)."""

import dataclasses

import pytest

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


def test_temp_schedule_config_is_frozen() -> None:
    """TempScheduleConfig is a frozen dataclass with the documented defaults."""
    cfg = TempScheduleConfig()
    assert cfg.kind == "exponential"
    assert cfg.tau_start == 1.0
    assert cfg.tau_end == 0.1
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.tau_start = 2.0  # ty: ignore[invalid-assignment]


def test_temp_schedule_config_rejects_non_positive_tau_start() -> None:
    """tau_start must be strictly positive (spec §3 [R16])."""
    with pytest.raises(ValueError, match="tau_start"):
        TempScheduleConfig(tau_start=0.0)


def test_temp_schedule_config_rejects_non_positive_tau_end() -> None:
    """tau_end must be strictly positive (spec §3 [R16])."""
    with pytest.raises(ValueError, match="tau_end"):
        TempScheduleConfig(tau_end=-0.5)


def test_model_config_defaults() -> None:
    cfg = ModelConfig(input_dim=784)
    assert cfg.input_dim == 784
    assert cfg.hidden_dims == (256, 256)
    assert cfg.output_dim == 10


def test_optim_config_defaults() -> None:
    cfg = OptimConfig()
    assert cfg.lr == 1e-3
    assert cfg.weight_decay == 0.0
    assert cfg.name == "adam"


def test_reg_config_defaults() -> None:
    cfg = RegConfig()
    assert cfg.lam == 0.0
    assert cfg.entropy_eps == 1e-6


def test_data_config_defaults() -> None:
    cfg = DataConfig()
    assert cfg.dataset == "mnist"
    assert cfg.batch_size == 128
    assert cfg.eval_batch_size == 512
    assert cfg.val_fraction == 0.1
    assert cfg.split_seed == 0
    assert cfg.drop_last is False
    assert cfg.data_dir == "data"
    assert cfg.num_workers == 0


def test_data_config_rejects_out_of_range_val_fraction() -> None:
    """val_fraction must be strictly inside (0, 1): both splits non-empty."""
    with pytest.raises(ValueError, match="val_fraction"):
        DataConfig(val_fraction=0.0)
    with pytest.raises(ValueError, match="val_fraction"):
        DataConfig(val_fraction=1.0)
    assert DataConfig(val_fraction=0.1).val_fraction == 0.1


def test_probe_config_defaults() -> None:
    cfg = ProbeConfig()
    assert cfg.num_neurons == 16
    assert cfg.seed == 0
    assert cfg.batch_size == 512


def test_checkpoint_config_defaults() -> None:
    cfg = CheckpointConfig()
    assert cfg.every_epochs == 1
    assert cfg.dir == "runs"


def test_surgery_config_defaults() -> None:
    cfg = SurgeryConfig()
    assert cfg.num_k == 21
    assert cfg.random_baseline is True
    assert cfg.tie_break == "identity"


def test_surgery_config_rejects_invalid_tie_break() -> None:
    """tie_break must be 'zero' or 'identity' (validated at construction)."""
    with pytest.raises(ValueError, match="tie_break"):
        SurgeryConfig(tie_break="bogus")
    assert SurgeryConfig(tie_break="zero").tie_break == "zero"
    assert SurgeryConfig(tie_break="identity").tie_break == "identity"


def test_surgery_config_rejects_num_k_below_two() -> None:
    """num_k must be >= 2 since make_k_grid divides by (num_k - 1)."""
    with pytest.raises(ValueError, match="num_k"):
        SurgeryConfig(num_k=1)
    with pytest.raises(ValueError, match="num_k"):
        SurgeryConfig(num_k=0)
    assert SurgeryConfig(num_k=2).num_k == 2


def test_wandb_config_defaults() -> None:
    cfg = WandbConfig()
    assert cfg.project == "kill-nonlinearities"
    assert cfg.entity is None
    assert cfg.mode == "online"
    assert cfg.group is None
    assert cfg.tags == ()


def test_train_config_defaults() -> None:
    cfg = TrainConfig()
    assert cfg.epochs == 20
    assert cfg.seed == 0
    assert cfg.device == "cpu"
    assert cfg.grad_clip is None


def test_experiment_config_composes_all() -> None:
    cfg = ExperimentConfig(name="demo", model=ModelConfig(input_dim=784))
    assert cfg.name == "demo"
    assert cfg.model.input_dim == 784
    assert isinstance(cfg.optim, OptimConfig)
    assert isinstance(cfg.temp_schedule, TempScheduleConfig)
    assert isinstance(cfg.reg, RegConfig)
    assert isinstance(cfg.data, DataConfig)
    assert isinstance(cfg.probe, ProbeConfig)
    assert isinstance(cfg.checkpoint, CheckpointConfig)
    assert isinstance(cfg.surgery, SurgeryConfig)
    assert isinstance(cfg.wandb, WandbConfig)
    assert isinstance(cfg.train, TrainConfig)
