"""Shared numerical-statistics helpers for the analysis pipeline.

Centralises the "constant series" guard that was previously duplicated as
``if np.std(x) == 0 or np.std(y) == 0: return 0.0`` across several modules
(issue #6).
"""

from __future__ import annotations

import numpy as np


def safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Return the Pearson correlation of *x* and *y*, or 0.0 when undefined.

    A constant (zero-variance) series makes the correlation mathematically
    undefined; callers treat that as "no evidence" rather than NaN.  The
    zero-variance guard above prevents the degenerate case, so corrcoef is
    always well-defined here.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2 or y.size < 2:
        return 0.0
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    corr = np.corrcoef(x, y)[0, 1]
    return float(corr) if np.isfinite(corr) else 0.0
