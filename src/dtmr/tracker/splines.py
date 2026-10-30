"""CPU-side spline interpolation matrix builder (node<->u), used once at
controller construction time."""
from __future__ import annotations

import numpy as np
from scipy.interpolate import make_interp_spline


def _build_interp_matrix_np(x: np.ndarray, x_new: np.ndarray) -> np.ndarray:
    """M (len(x_new), len(x)) such that M @ y samples a k=2 spline through (x, y)
    at x_new."""
    n = len(x)
    mat = np.zeros((len(x_new), n), dtype=np.float32)
    basis = np.eye(n, dtype=np.float64)
    for i in range(n):
        spline = make_interp_spline(x.astype(np.float64), basis[i], k=2)
        mat[:, i] = spline(x_new.astype(np.float64)).astype(np.float32)
    return mat

