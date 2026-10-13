"""Moving-block bootstrap confidence interval for mean trade return."""
from __future__ import annotations

import numpy as np


def block_bootstrap_ci(values, block_size: int = 5, n_boot: int = 5000,
                       alpha: float = 0.10, seed: int | None = 0) -> tuple[float, float]:
    """Return a percentile CI for mean returns using overlapping blocks.

    This preserves some local serial dependence. It does not correct every form
    of dependence and should not be treated as a guarantee of valid inference.
    """
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    if n_boot < 1:
        raise ValueError("n_boot must be >= 1")
    if len(x) == 0:
        return float("nan"), float("nan")
    b = min(max(int(block_size), 1), len(x))
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(x) - b + 1, size=(n_boot, int(np.ceil(len(x) / b))))
    indices = (starts[..., None] + np.arange(b)).reshape(n_boot, -1)[:, :len(x)]
    means = x[indices].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)
