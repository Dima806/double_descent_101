"""Train and test error versus model capacity, across the interpolation threshold.

One row per (seed, n_params, noise). Every noise level reuses the same inputs, the same standard
normal noise draw and the same features (common random numbers), so differences between noise
levels are due to the noise level alone.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

from src.arrays import FloatArray, IntArray
from src.datasets.synthetic import Task
from src.models.base import FeatureMap
from src.models.min_norm import min_norm_fit, ridge_fit

FeatureFactory = Callable[[np.random.Generator, int], FeatureMap]

COLUMNS = [
    "model",
    "capacity",
    "n_params",
    "n_train",
    "noise",
    "seed",
    "train_mse",
    "test_mse",
    "coef_norm",
]


def capacity_sweep(
    factory: FeatureFactory,
    task: Task,
    n_train: int,
    noise_levels: Sequence[float] | FloatArray,
    n_params_grid: Sequence[int] | IntArray,
    n_seeds: int,
    seed: int,
    ridge_lambda: float = 0.0,
) -> pd.DataFrame:
    """Min-norm fits for every model size in ``n_params_grid`` on ``n_seeds`` training draws.

    Test MSE is measured against the noiseless truth on the task's fixed test inputs.
    ``ridge_lambda > 0`` switches to ridge with penalty ``ridge_lambda`` per training point
    (minimise (1/n)||y - phi theta||^2 + lambda ||theta||^2); 0 is the min-norm ("ridgeless") fit.
    Returns a tidy DataFrame with ``COLUMNS``.
    """
    grid = np.asarray(sorted(set(int(p) for p in n_params_grid)), dtype=int)
    noises = np.asarray(noise_levels, dtype=np.float64)
    x_test = task.test_x()
    f_test = task.f(x_test)
    records: list[dict[str, float | int | str]] = []
    for s, rng in enumerate(np.random.default_rng(seed).spawn(n_seeds)):
        x = task.sample_x(n_train, rng)
        f_train = task.f(x)
        eps = rng.standard_normal(n_train)
        y = f_train[:, None] + eps[:, None] * noises[None, :]  # (n_train, n_noise)
        fmap = factory(rng, int(grid.max()))
        max_cap = fmap.capacity_for(int(grid.max()))
        phi_train = fmap(x, max_cap)
        phi_test = fmap(x_test, max_cap)
        for p in grid:
            if ridge_lambda > 0:
                theta = ridge_fit(phi_train[:, :p], y, ridge_lambda * n_train)
            else:
                theta = min_norm_fit(phi_train[:, :p], y)
            train_mse = np.mean((phi_train[:, :p] @ theta - y) ** 2, axis=0)
            test_mse = np.mean((phi_test[:, :p] @ theta - f_test[:, None]) ** 2, axis=0)
            norms = np.linalg.norm(theta, axis=0)
            for j, sigma in enumerate(noises):
                records.append(
                    {
                        "model": fmap.name,
                        "capacity": fmap.capacity_for(int(p)),
                        "n_params": int(p),
                        "n_train": n_train,
                        "noise": float(sigma),
                        "seed": s,
                        "train_mse": float(train_mse[j]),
                        "test_mse": float(test_mse[j]),
                        "coef_norm": float(norms[j]),
                    }
                )
    return pd.DataFrame.from_records(records, columns=COLUMNS)
