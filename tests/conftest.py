"""Shared pytest configuration and Hypothesis profiles.

The active Hypothesis profile is chosen by the ``HYPOTHESIS_PROFILE`` environment
variable and defaults to ``dev`` (fast). CI should export ``HYPOTHESIS_PROFILE=ci``
for a more thorough search. See docs/development/testing.md.
"""

import os

from hypothesis import settings

settings.register_profile("dev", max_examples=25)
settings.register_profile("ci", max_examples=500, deadline=None)
settings.load_profile(os.getenv("HYPOTHESIS_PROFILE", "dev"))
