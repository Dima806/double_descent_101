"""Common interface for capacity-indexed feature maps."""

from __future__ import annotations

from typing import Protocol

from src.arrays import FloatArray


class FeatureMap(Protocol):
    """Maps inputs to a design matrix whose width is set by an integer capacity.

    Feature maps here are NESTED: the features at capacity c are the first ``n_params(c)``
    columns of the features at any larger capacity. Sweeps rely on this to build the widest
    design matrix once and slice it.
    """

    @property
    def name(self) -> str:
        """Model family name used in result tables."""
        ...

    def __call__(self, x: FloatArray, capacity: int) -> FloatArray:
        """Design matrix of shape (len(x), n_params(capacity))."""
        ...

    def n_params(self, capacity: int) -> int:
        """Number of fitted coefficients, intercept included."""
        ...

    def capacity_for(self, n_params: int) -> int:
        """Inverse of ``n_params``."""
        ...
