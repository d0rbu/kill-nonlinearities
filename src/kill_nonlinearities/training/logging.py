"""Logger protocol and concrete loggers (spec §4.6).

``wandb`` is imported **lazily inside ``WandbLogger`` only**, so importing the
pure loggers (``NullLogger``/``InMemoryLogger``) never requires wandb.
"""

from collections.abc import Mapping
from pathlib import Path
from typing import Protocol, runtime_checkable


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
