"""Test error versus training-set size at a FIXED model size ("more data can hurt").

With n_params fixed at p, growing n_train moves the ratio p / n_train down through 1, so the
model walks INTO the interpolation peak from the overparameterized side before walking out.
Training sets are nested within a seed (the first n points of one pool), so curves are smooth.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

from src.arrays import FloatArray, IntArray
from src.datasets.synthetic import Task
from src.evaluation.capacity_sweep import COLUMNS
from src.models.base import FeatureMap
from src.models.min_norm import min_norm_fit

FeatureFactory = Callable[[np.random.Generator, int], FeatureMap]


def data_sweep(
    factory: FeatureFactory,
    task: Task,
    n_params: int,
    n_train_grid: Sequence[int] | IntArray,
    noise_levels: Sequence[float] | FloatArray,
    n_seeds: int,
    seed: int,
) -> pd.DataFrame:
    """Min-norm fits of a fixed-size model on growing training sets. Same columns as sweeps."""
    sizes = np.asarray(sorted(set(int(n) for n in n_train_grid)), dtype=int)
    noises = np.asarray(noise_levels, dtype=np.float64)
    x_test = task.test_x()
    f_test = task.f(x_test)
    n_max = int(sizes.max())
    records: list[dict[str, float | int | str]] = []
    for s, rng in enumerate(np.random.default_rng(seed).spawn(n_seeds)):
        x_pool = task.sample_x(n_max, rng)
        f_pool = task.f(x_pool)
        eps = rng.standard_normal(n_max)
        y_pool = f_pool[:, None] + eps[:, None] * noises[None, :]
        fmap = factory(rng, n_params)
        cap = fmap.capacity_for(n_params)
        phi_pool = fmap(x_pool, cap)
        phi_test = fmap(x_test, cap)
        for n in sizes:
            theta = min_norm_fit(phi_pool[:n], y_pool[:n])
            train_mse = np.mean((phi_pool[:n] @ theta - y_pool[:n]) ** 2, axis=0)
            test_mse = np.mean((phi_test @ theta - f_test[:, None]) ** 2, axis=0)
            norms = np.linalg.norm(theta, axis=0)
            for j, sigma in enumerate(noises):
                records.append(
                    {
                        "model": fmap.name,
                        "capacity": cap,
                        "n_params": n_params,
                        "n_train": int(n),
                        "noise": float(sigma),
                        "seed": s,
                        "train_mse": float(train_mse[j]),
                        "test_mse": float(test_mse[j]),
                        "coef_norm": float(norms[j]),
                    }
                )
    return pd.DataFrame.from_records(records, columns=COLUMNS)


def error_surface(
    factory: FeatureFactory,
    task: Task,
    n_params_grid: Sequence[int] | IntArray,
    n_train_grid: Sequence[int] | IntArray,
    noise: float,
    n_seeds: int,
    seed: int,
) -> pd.DataFrame:
    """Test MSE over a (n_params, n_train) grid: the peak traces the diagonal n_params == n_train.

    Returns columns: n_params, n_train, seed, test_mse.
    """
    ps = np.asarray(sorted(set(int(p) for p in n_params_grid)), dtype=int)
    ns = np.asarray(sorted(set(int(n) for n in n_train_grid)), dtype=int)
    x_test = task.test_x()
    f_test = task.f(x_test)
    records: list[dict[str, float | int]] = []
    for s, rng in enumerate(np.random.default_rng(seed).spawn(n_seeds)):
        x_pool = task.sample_x(int(ns.max()), rng)
        y_pool = task.f(x_pool) + noise * rng.standard_normal(int(ns.max()))
        fmap = factory(rng, int(ps.max()))
        cap = fmap.capacity_for(int(ps.max()))
        phi_pool = fmap(x_pool, cap)
        phi_test = fmap(x_test, cap)
        for n in ns:
            for p in ps:
                theta = min_norm_fit(phi_pool[:n, :p], y_pool[:n])
                err = float(np.mean((phi_test[:, :p] @ theta - f_test) ** 2))
                records.append({"n_params": int(p), "n_train": int(n), "seed": s, "test_mse": err})
    return pd.DataFrame.from_records(records)
