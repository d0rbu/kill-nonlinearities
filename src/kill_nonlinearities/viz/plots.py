"""Static figures for the Phase 1a pipeline (spec §4.12, §5).

Headless rendering rule (spec §4.12, [R10]): select the non-interactive Agg
backend *before* importing pyplot; never call ``plt.show``; always ``savefig``
then ``plt.close(fig)``. Every public function takes a full FILE path and returns
the saved figure ``Path``.
"""

import matplotlib

matplotlib.use("Agg")  # must precede the pyplot import below (spec §4.12)

__all__: list[str] = []
