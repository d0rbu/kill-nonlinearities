"""Unit tests for kill_nonlinearities.viz.animation (imageio + pillow GIFs)."""

from pathlib import Path

import torch

from kill_nonlinearities.analysis.statistics import FrameStats
from kill_nonlinearities.viz import animation


def _toy_frames() -> list[FrameStats]:
    frames: list[FrameStats] = []
    for step in (0, 1, 2):
        frames.append(
            FrameStats(
                step=step,
                probe_activations={
                    ("relu0", 0): torch.randn(32),
                    ("relu0", 1): torch.randn(32),
                    ("relu1", 0): torch.randn(32),
                },
                q_by_site={
                    "relu0": torch.rand(4),
                    "relu1": torch.rand(4),
                },
            )
        )
    return frames


def test_render_activation_gif_writes_nonempty_file(tmp_path: Path) -> None:
    """render_activation_gif writes a non-empty .gif and returns its Path (spec §5)."""
    out = tmp_path / "activation.gif"
    result = animation.render_activation_gif(_toy_frames(), out)

    assert result == out
    assert out.exists()
    assert out.stat().st_size > 0


def test_render_activation_gif_frame_count_via_helper(tmp_path: Path) -> None:
    """gif_frame_count returns one frame per FrameStats (spec §7 frame_count)."""
    frames = _toy_frames()
    out = tmp_path / "activation.gif"
    animation.render_activation_gif(frames, out)

    assert animation.gif_frame_count(out) == len(frames)


def test_render_qi_bimodality_gif_writes_nonempty_file(tmp_path: Path) -> None:
    """render_qi_bimodality_gif writes a non-empty .gif and returns its Path (§5)."""
    out = tmp_path / "qi_bimodality.gif"
    result = animation.render_qi_bimodality_gif(_toy_frames(), out)

    assert result == out
    assert out.exists()
    assert out.stat().st_size > 0


def test_render_qi_bimodality_gif_frame_count_via_helper(tmp_path: Path) -> None:
    """gif_frame_count returns one frame per FrameStats for the q_i gif (§7)."""
    frames = _toy_frames()
    out = tmp_path / "qi_bimodality.gif"
    animation.render_qi_bimodality_gif(frames, out)

    assert animation.gif_frame_count(out) == len(frames)
