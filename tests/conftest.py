"""Shared pytest configuration and Hypothesis profiles.

The active Hypothesis profile is chosen by the ``HYPOTHESIS_PROFILE`` environment
variable and defaults to ``dev`` (fast). CI should export ``HYPOTHESIS_PROFILE=ci``
for a more thorough search. See docs/development/testing.md.
"""

import os

from hypothesis import settings

# Keep the test suite fully offline and headless (spec §7, §8): never hit the
# wandb backend and force matplotlib's non-interactive Agg backend.
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("MPLBACKEND", "Agg")

settings.register_profile("dev", max_examples=25)
settings.register_profile("ci", max_examples=500, deadline=None)
settings.load_profile(os.getenv("HYPOTHESIS_PROFILE", "dev"))
