"""Polynomial regression in the Legendre basis, indexed by degree.

Raw monomials x**k give a Vandermonde matrix whose conditioning explodes by degree ~15-20, and
float64 round-off then fakes or hides the interpolation peak. Legendre polynomials are orthogonal
on [-1, 1], so the design matrix stays well conditioned to high degree. The basis also fixes WHICH
interpolant is minimum-norm: with ``decay > 0`` high-order polynomials are scaled down, so the
min-norm solution prefers smooth (low-order) explanations of the data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.polynomial import legendre

from src.arrays import FloatArray


@dataclass(frozen=True)
class LegendreFeatures:
    """Feature map x -> [s_0 P_0(x), ..., s_d P_d(x)]; n_params = degree + 1."""

    normalize: bool = True
    decay: float = 0.0
    name: str = "polynomial"

    def scales(self, degree: int) -> FloatArray:
        k = np.arange(degree + 1, dtype=np.float64)
        s = np.sqrt(2.0 * k + 1.0) if self.normalize else np.ones_like(k)
        return s / (1.0 + k) ** self.decay

    def __call__(self, x: FloatArray, capacity: int) -> FloatArray:
        if capacity < 0:
            raise ValueError("degree must be >= 0")
        return legendre.legvander(np.asarray(x, dtype=np.float64), capacity) * self.scales(
            capacity
        )

    def n_params(self, capacity: int) -> int:
        return capacity + 1

    def capacity_for(self, n_params: int) -> int:
        return n_params - 1


def monomial_features(x: FloatArray, degree: int) -> FloatArray:
    """Raw Vandermonde matrix [1, x, ..., x^degree]. Kept only to show its ill-conditioning."""
    return np.vander(np.asarray(x, dtype=np.float64), degree + 1, increasing=True)
