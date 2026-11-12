"""Random ReLU features: the simplest overparameterized model.

phi_j(x) = max(0, <w_j, x> + b_j) with random, frozen (w_j, b_j); only the linear read-out is
fitted. Equivalent to a two-layer network whose first layer is never trained.
Width N gives n_params = N (the bias lives inside the ReLU; no separate intercept column).
Weights are w_j ~ N(0, weight_scale^2 / d) so <w_j, x> has unit scale for x ~ N(0, I_d).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.arrays import FloatArray


def _as_2d(x: FloatArray) -> FloatArray:
    arr = np.asarray(x, dtype=np.float64)
    return arr[:, None] if arr.ndim == 1 else arr


@dataclass(frozen=True)
class RandomReLUFeatures:
    """Frozen draw of ``max_width`` ReLU features; width N uses the first N (nested widths)."""

    weights: FloatArray  # (max_width, dim)
    biases: FloatArray  # (max_width,)
    name: str = "random_features"

    @property
    def max_width(self) -> int:
        return int(self.weights.shape[0])

    def __call__(self, x: FloatArray, capacity: int) -> FloatArray:
        if not 1 <= capacity <= self.max_width:
            raise ValueError(f"width must be in [1, {self.max_width}], got {capacity}")
        z = _as_2d(x) @ self.weights[:capacity].T
        return np.maximum(z + self.biases[:capacity], 0.0)

    def n_params(self, capacity: int) -> int:
        return capacity

    def capacity_for(self, n_params: int) -> int:
        return n_params


def draw_random_relu(
    max_width: int,
    rng: np.random.Generator,
    dim: int = 1,
    weight_scale: float = 1.0,
    bias_scale: float = 1.0,
) -> RandomReLUFeatures:
    """Draw w ~ N(0, weight_scale^2 / dim) and b ~ U(-bias_scale, bias_scale) once; reuse them."""
    weights = (weight_scale / np.sqrt(dim)) * rng.standard_normal((max_width, dim))
    biases = rng.uniform(-bias_scale, bias_scale, size=max_width)
    return RandomReLUFeatures(weights=weights, biases=biases)
