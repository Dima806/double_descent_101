"""The interpolation threshold: the capacity at which n_params == n_train.

Below it (underparameterized) least squares cannot fit every training point. At it, there is
exactly one interpolating solution and it must thread every noisy point. Above it
(overparameterized) there are infinitely many interpolants and the min-norm one is chosen.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Literal

import numpy as np
import pandas as pd

from src.arrays import IntArray
from src.datasets.synthetic import Task
from src.models.base import FeatureMap

Regime = Literal["underparameterized", "threshold", "overparameterized"]


def interpolation_threshold(n_train: int) -> int:
    """Number of parameters at which a generic linear model first interpolates n_train points."""
    if n_train < 1:
        raise ValueError("n_train must be >= 1")
    return n_train


def param_ratio(n_params: int | float, n_train: int) -> float:
    """gamma = n_params / n_train; 1 marks the threshold."""
    return float(n_params) / float(n_train)


def regime(n_params: int, n_train: int) -> Regime:
    """Label a model size relative to the threshold."""
    if n_params < n_train:
        return "underparameterized"
    if n_params == n_train:
        return "threshold"
    return "overparameterized"


def capacity_grid(
    n_train: int,
    max_ratio: float,
    n_under: int | None = None,
    dense_window: float = 0.15,
    n_over: int = 20,
) -> np.ndarray:
    """Sorted unique n_params values spanning 1 .. max_ratio * n_train.

    Always contains n_train exactly and is dense within +-dense_window of it (a grid that steps
    over the threshold blurs or misses the peak). ``n_under=None`` uses every integer below the
    dense window; otherwise that many evenly spaced values. Above the window: ``n_over``
    log-spaced values.
    """
    if max_ratio <= 1.0:
        raise ValueError("max_ratio must exceed 1 to cross the threshold")
    lo = max(1, int(np.floor((1.0 - dense_window) * n_train)))
    hi = max(n_train + 1, int(np.ceil((1.0 + dense_window) * n_train)))
    if n_under is None:
        under = np.arange(1, lo)
    else:
        under = np.unique(np.linspace(1, lo, n_under).round().astype(int))
    step = max(1, n_train // 40)
    dense = np.unique(np.concatenate([np.arange(lo, hi + 1, step), [n_train, hi]]))
    top = max(hi + 1, int(round(max_ratio * n_train)))
    over = np.unique(np.geomspace(hi + 1, top, n_over).round().astype(int))
    grid = np.unique(np.concatenate([under, dense, over])).astype(int)
    return grid[grid >= 1]


def conditioning_sweep(
    factory: Callable[[np.random.Generator, int], FeatureMap],
    task: Task,
    n_train: int,
    n_params_grid: Sequence[int] | IntArray,
    n_seeds: int,
    seed: int,
) -> pd.DataFrame:
    """Smallest singular value of the training design matrix at each model size.

    The fitted noise is amplified by roughly 1 / sigma_min. At p == n the matrix is square and
    sigma_min collapses towards zero, which is why the error spikes there; on either side it
    recovers. Draws match ``capacity_sweep`` with the same seed.
    Returns columns: n_params, seed, sigma_min.
    """
    grid = np.asarray(sorted(set(int(p) for p in n_params_grid)), dtype=int)
    records: list[dict[str, float | int]] = []
    for s, rng in enumerate(np.random.default_rng(seed).spawn(n_seeds)):
        x = task.sample_x(n_train, rng)
        rng.standard_normal(n_train)  # the label-noise draw, kept so draws match the sweeps
        fmap = factory(rng, int(grid.max()))
        phi = fmap(x, fmap.capacity_for(int(grid.max())))
        for p in grid:
            sv = np.linalg.svd(phi[:, :p], compute_uv=False)
            records.append({"n_params": int(p), "seed": s, "sigma_min": float(sv[-1])})
    return pd.DataFrame.from_records(records)
