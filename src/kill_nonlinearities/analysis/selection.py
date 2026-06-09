"""Surgery selection: ranking, k-grid, top-k, and mode assignment (spec §4.10)."""


def make_k_grid(total: int, num_k: int) -> list[int]:
    """Target ``num_k`` evenly-spaced k-values in ``[0, total]`` (spec §4.10).

    Always includes ``0`` and ``total``. The realized length equals the number
    of UNIQUE rounded points, which may be fewer than ``num_k`` on tiny models.
    """
    return sorted({round(i * total / (num_k - 1)) for i in range(num_k)})
