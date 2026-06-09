"""Logger protocol and concrete loggers (spec §4.6).

``wandb`` is imported **lazily inside ``WandbLogger`` only**, so importing the
pure loggers (``NullLogger``/``InMemoryLogger``) never requires wandb.
"""

from collections.abc import Mapping
from pathlib import Path
from typing import Protocol, runtime_checkable

from kill_nonlinearities.config import WandbConfig


@runtime_checkable
class Logger(Protocol):
    """Sink for scalars, images, videos, config, and run lifecycle."""

    def log_scalars(self, values: Mapping[str, float], step: int) -> None: ...

    def log_image(self, name: str, path: Path, step: int | None = None) -> None: ...

    def log_video(self, name: str, path: Path, step: int | None = None) -> None: ...

    def log_config(self, config: Mapping[str, object]) -> None: ...

    def finish(self) -> None: ...


class NullLogger:
    """No-op logger; the default in library code."""

    def log_scalars(self, values: Mapping[str, float], step: int) -> None:
        return None

    def log_image(self, name: str, path: Path, step: int | None = None) -> None:
        return None

    def log_video(self, name: str, path: Path, step: int | None = None) -> None:
        return None

    def log_config(self, config: Mapping[str, object]) -> None:
        return None

    def finish(self) -> None:
        return None


class InMemoryLogger:
    """Records everything to inspectable attributes; used by tests."""

    def __init__(self) -> None:
        self.scalars: list[tuple[int, dict[str, float]]] = []
        self.images: list[tuple[str, Path, int | None]] = []
        self.videos: list[tuple[str, Path, int | None]] = []
        self.config: dict[str, object] = {}
        self.finished: bool = False

    def log_scalars(self, values: Mapping[str, float], step: int) -> None:
        self.scalars.append((step, dict(values)))

    def log_image(self, name: str, path: Path, step: int | None = None) -> None:
        self.images.append((name, path, step))

    def log_video(self, name: str, path: Path, step: int | None = None) -> None:
        self.videos.append((name, path, step))

    def log_config(self, config: Mapping[str, object]) -> None:
        self.config = dict(config)

    def finish(self) -> None:
        self.finished = True


class WandbLogger:
    """Logger that streams to Weights & Biases.

    ``wandb`` is imported lazily inside the methods so importing this module
    never requires wandb. When ``run is None`` a new run is started via
    ``wandb.init``; otherwise this logger **attaches** to the already-active
    ``run`` (no second ``wandb.init``). Not unit-tested (spec §7).
    """

    def __init__(self, wandb_config: WandbConfig, run: object | None = None):
        self._config = wandb_config
        self._run = run

    def _ensure_run(self) -> object:  # pragma: no cover - network glue (spec §7)
        if self._run is None:
            import wandb

            self._run = wandb.init(
                project=self._config.project,
                entity=self._config.entity,
                mode=self._config.mode,  # ty: ignore[invalid-argument-type]
                group=self._config.group,
                tags=list(self._config.tags),
            )
        return self._run

    def log_scalars(
        self, values: Mapping[str, float], step: int
    ) -> None:  # pragma: no cover - network glue (spec §7)
        self._ensure_run()
        import wandb

        wandb.log(dict(values), step=step)

    def log_image(
        self, name: str, path: Path, step: int | None = None
    ) -> None:  # pragma: no cover - network glue (spec §7)
        self._ensure_run()
        import wandb

        wandb.log({name: wandb.Image(str(path))}, step=step)

    def log_video(
        self, name: str, path: Path, step: int | None = None
    ) -> None:  # pragma: no cover - network glue (spec §7)
        self._ensure_run()
        import wandb

        wandb.log({name: wandb.Video(str(path))}, step=step)

    def log_config(
        self, config: Mapping[str, object]
    ) -> None:  # pragma: no cover - network glue (spec §7)
        run = self._ensure_run()
        run.config.update(dict(config))  # ty: ignore[unresolved-attribute]

    def finish(self) -> None:  # pragma: no cover - network glue (spec §7)
        if self._run is not None:
            import wandb

            wandb.finish()
            self._run = None
