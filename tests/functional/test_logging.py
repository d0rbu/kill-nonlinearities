"""Functional tests for the logging protocol implementations (spec §4.6)."""

from pathlib import Path

from kill_nonlinearities.training.logging import (
    InMemoryLogger,
    Logger,
    NullLogger,
)


def test_null_logger_is_a_logger_and_no_ops(tmp_path: Path) -> None:
    logger: Logger = NullLogger()
    # None of these should raise or return anything meaningful.
    logger.log_scalars({"loss/task": 1.0}, step=0)
    logger.log_image("fig", tmp_path / "fig.png", step=1)
    logger.log_video("gif", tmp_path / "anim.gif", step=2)
    logger.log_config({"lr": 1e-3})
    logger.finish()


def test_in_memory_logger_records_scalars_with_steps() -> None:
    logger = InMemoryLogger()
    logger.log_scalars({"loss/task": 1.5, "tau": 0.5}, step=0)
    logger.log_scalars({"loss/task": 1.0, "tau": 0.4}, step=1)
    assert logger.scalars == [
        (0, {"loss/task": 1.5, "tau": 0.5}),
        (1, {"loss/task": 1.0, "tau": 0.4}),
    ]


def test_in_memory_logger_records_images_and_videos(tmp_path: Path) -> None:
    logger = InMemoryLogger()
    img = tmp_path / "loss.png"
    vid = tmp_path / "anim.gif"
    logger.log_image("loss_curves", img, step=3)
    logger.log_video("activation_gif", vid, step=None)
    assert logger.images == [("loss_curves", img, 3)]
    assert logger.videos == [("activation_gif", vid, None)]


def test_in_memory_logger_records_config_and_finish() -> None:
    logger = InMemoryLogger()
    logger.log_config({"lr": 1e-3, "lam": 0.5})
    assert logger.config == {"lr": 1e-3, "lam": 0.5}
    assert logger.finished is False
    logger.finish()
    assert logger.finished is True
