"""Animated GIFs for the Phase 1a pipeline (spec §4.12, §5).

Built via imageio's pillow plugin from per-FrameStats frames. The same headless
Agg rule as plots.py applies: select Agg before importing pyplot; never show;
close every figure. Public renderers take a full file ``path`` and return it.
"""

from collections.abc import Sequence
from pathlib import Path

import imageio.v3 as iio
import matplotlib

matplotlib.use("Agg")  # must precede the pyplot import below (spec §4.12)

import matplotlib.pyplot as plt
import numpy as np

from kill_nonlinearities.analysis.statistics import FrameStats

__all__ = ["gif_frame_count", "render_activation_gif"]


def _figure_to_rgb(fig: plt.Figure) -> np.ndarray:
    """Render an Agg figure to an [H, W, 3] uint8 array (no temp files)."""
    fig.canvas.draw()
    # buffer_rgba is Agg-specific (forced above); base-class type lacks it.
    rgba = np.asarray(fig.canvas.buffer_rgba())  # ty: ignore[unresolved-attribute]
    return rgba[..., :3].copy()


def gif_frame_count(path: Path) -> int:
    """Number of frames in the gif at ``path`` (spec §4.12 / §7)."""
    return int(iio.imread(path, index=None).shape[0])


def render_activation_gif(frames: Sequence[FrameStats], path: Path) -> Path:
    """Animate the probe-neuron pre-activation histograms over checkpoints (spec §5).

    Probe neurons are inferred from ``frames[0].probe_activations.keys()`` and held
    in a fixed (site, index) layout so the animation reflects weight evolution only.
    One gif frame per ``FrameStats``. Takes a full file ``path`` and returns it.
    """
    neuron_keys = sorted(frames[0].probe_activations.keys())
    images: list[np.ndarray] = []
    for frame in frames:
        fig, axes = plt.subplots(
            1,
            len(neuron_keys),
            squeeze=False,
            figsize=(3 * len(neuron_keys), 3),
        )
        for ax, key in zip(axes[0], neuron_keys, strict=True):
            values = frame.probe_activations[key].detach().cpu().tolist()
            ax.hist(values, bins=20)
            ax.set_title(f"{key[0]}[{key[1]}]")
            ax.set_xlabel("pre-activation")
        fig.suptitle(f"step {frame.step}")
        fig.tight_layout()
        images.append(_figure_to_rgb(fig))
        plt.close(fig)
    iio.imwrite(path, np.stack(images), plugin="pillow", extension=".gif")
    return path
