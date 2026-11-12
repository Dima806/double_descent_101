"""Empirical bias-variance decomposition over many independent training draws.

For each model size, refit on R fresh training sets (fresh inputs, fresh noise and, for random
features, fresh feature weights) and evaluate on the fixed test inputs:

    bias^2   = mean_x ( mean_r f_r(x) - f(x) )^2
    variance = mean_x   var_r  f_r(x)              (ddof = 0)
    expected excess MSE = mean_r mean_x (f_r(x) - f(x))^2 = bias^2 + variance   (exact identity)

Test error here is against the noiseless truth; add sigma^2 for error against noisy labels.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

from src.arrays import IntArray
from src.datasets.synthetic import Task
from src.models.base import FeatureMap
from src.models.min_norm import min_norm_fit

FeatureFactory = Callable[[np.random.Generator, int], FeatureMap]


def bias_variance(
    factory: FeatureFactory,
    task: Task,
    n_train: int,
    noise: float,
    n_params_grid: Sequence[int] | IntArray,
    n_draws: int,
    seed: int,
) -> pd.DataFrame:
    """Bias^2, variance and expected excess MSE per model size.

    Returns columns: n_params, n_train, noise, bias2, variance, expected_mse, noise_var.
    """
    grid = np.asarray(sorted(set(int(p) for p in n_params_grid)), dtype=int)
    x_test = task.test_x()
    f_test = task.f(x_test)
    n_test = f_test.shape[0]
    s1 = np.zeros((grid.size, n_test))
    s2 = np.zeros((grid.size, n_test))
    sq_err = np.zeros(grid.size)
    for rng in np.random.default_rng(seed).spawn(n_draws):
        x = task.sample_x(n_train, rng)
        y = task.f(x) + noise * rng.standard_normal(n_train)
        fmap = factory(rng, int(grid.max()))
        max_cap = fmap.capacity_for(int(grid.max()))
        phi_train = fmap(x, max_cap)
        phi_test = fmap(x_test, max_cap)
        for i, p in enumerate(grid):
            theta = min_norm_fit(phi_train[:, :p], y)
            pred = phi_test[:, :p] @ theta
            s1[i] += pred
            s2[i] += pred**2
            sq_err[i] += np.mean((pred - f_test) ** 2)
    mean_pred = s1 / n_draws
    variance = np.maximum(s2 / n_draws - mean_pred**2, 0.0).mean(axis=1)
    bias2 = ((mean_pred - f_test) ** 2).mean(axis=1)
    return pd.DataFrame(
        {
            "n_params": grid,
            "n_train": n_train,
            "noise": noise,
            "bias2": bias2,
            "variance": variance,
            "expected_mse": sq_err / n_draws,
            "noise_var": noise**2,
        }
    )


def signal_noise_split(
    factory: FeatureFactory,
    task: Task,
    n_train: int,
    noise: float,
    n_params_grid: Sequence[int] | IntArray,
    n_seeds: int,
    seed: int,
) -> pd.DataFrame:
    """Split each min-norm fit into the part driven by the signal and the part driven by noise.

    The min-norm fit is linear in y = f_train + noise * eps, so theta = theta_signal + theta_noise
    with theta_signal fitted to f_train and theta_noise fitted to noise * eps. On the training
    points theta_noise reproduces the noise exactly once p >= n (memorisation). Benign
    overfitting means its footprint on TEST points, ``noise_energy``, is small.

    Draws match ``capacity_sweep`` with the same seed, so ``total`` equals its test_mse.
    Returns columns: n_params, seed, signal_error, noise_energy, total, train_mse.
    """
    grid = np.asarray(sorted(set(int(p) for p in n_params_grid)), dtype=int)
    x_test = task.test_x()
    f_test = task.f(x_test)
    records: list[dict[str, float | int]] = []
    for s, rng in enumerate(np.random.default_rng(seed).spawn(n_seeds)):
        x = task.sample_x(n_train, rng)
        f_train = task.f(x)
        eps = rng.standard_normal(n_train)
        fmap = factory(rng, int(grid.max()))
        max_cap = fmap.capacity_for(int(grid.max()))
        phi_train = fmap(x, max_cap)
        phi_test = fmap(x_test, max_cap)
        targets = np.column_stack([f_train, noise * eps])
        for p in grid:
            theta = min_norm_fit(phi_train[:, :p], targets)  # (p, 2): signal, noise
            test_parts = phi_test[:, :p] @ theta
            train_fit = phi_train[:, :p] @ theta.sum(axis=1)
            records.append(
                {
                    "n_params": int(p),
                    "seed": s,
                    "signal_error": float(np.mean((test_parts[:, 0] - f_test) ** 2)),
                    "noise_energy": float(np.mean(test_parts[:, 1] ** 2)),
                    "total": float(np.mean((test_parts.sum(axis=1) - f_test) ** 2)),
                    "train_mse": float(np.mean((train_fit - (f_train + noise * eps)) ** 2)),
                }
            )
    return pd.DataFrame.from_records(records)
