"""Theory helpers: the interpolation threshold and the bias-variance decomposition."""

from src.theory.decomposition import bias_variance, signal_noise_split
from src.theory.threshold import (
    capacity_grid,
    conditioning_sweep,
    interpolation_threshold,
    param_ratio,
    regime,
)

__all__ = [
    "bias_variance",
    "capacity_grid",
    "conditioning_sweep",
    "interpolation_threshold",
    "param_ratio",
    "regime",
    "signal_noise_split",
]
