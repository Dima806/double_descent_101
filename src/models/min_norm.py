"""Minimum-norm least squares via the SVD: the solver behind every linear model here.

For p < n (underparameterized) it is ordinary least squares. For p >= n there are infinitely
many coefficient vectors that fit the training data exactly; the pseudo-inverse picks the one
with the smallest Euclidean norm, which lies in the row space of the design matrix. That choice
of interpolant is what makes the second descent happen.
"""

from __future__ import annotations

import numpy as np

from src.arrays import FloatArray


def default_rtol(n: int, p: int) -> float:
    """Machine-precision singular-value cutoff, relative to the largest singular value.

    Matches ``np.linalg.lstsq(rcond=None)``. A larger cutoff truncates small singular values,
    which is implicit regularization and flattens the interpolation peak.
    """
    return float(np.finfo(np.float64).eps * max(n, p))


def min_norm_fit(phi: FloatArray, y: FloatArray, rtol: float | None = None) -> FloatArray:
    """Minimum-norm least-squares coefficients.

    phi: (n, p) design matrix. y: (n,) or (n, k) targets (k columns solved at once).
    Returns theta with shape (p,) or (p, k).
    """
    n, p = phi.shape
    u, s, vt = np.linalg.svd(phi, full_matrices=False)
    tol = (default_rtol(n, p) if rtol is None else rtol) * (s[0] if s.size else 0.0)
    keep = s > tol
    s_inv = np.zeros_like(s)
    s_inv[keep] = 1.0 / s[keep]
    uty = u.T @ y
    scaled = s_inv[:, None] * uty if uty.ndim == 2 else s_inv * uty
    return vt.T @ scaled


def ridge_fit(phi: FloatArray, y: FloatArray, lam: float) -> FloatArray:
    """Ridge coefficients minimising ||phi @ theta - y||^2 + lam * ||theta||^2 (lam >= 0).

    lam -> 0+ recovers ``min_norm_fit`` (the "ridgeless" limit). Shapes as in ``min_norm_fit``.
    """
    if lam < 0:
        raise ValueError("lam must be non-negative")
    if lam == 0:
        return min_norm_fit(phi, y)
    u, s, vt = np.linalg.svd(phi, full_matrices=False)
    filt = s / (s**2 + lam)
    uty = u.T @ y
    scaled = filt[:, None] * uty if uty.ndim == 2 else filt * uty
    return vt.T @ scaled


def predict(phi: FloatArray, theta: FloatArray) -> FloatArray:
    """Predictions phi @ theta. Shape (n,) or (n, k)."""
    return phi @ theta
