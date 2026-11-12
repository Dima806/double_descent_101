"""Harness: build tasks and models from settings, run sweeps, aggregate and summarise curves."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
import pandas as pd

from src.arrays import FloatArray, IntArray
from src.config import Settings
from src.datasets.synthetic import Task
from src.evaluation.capacity_sweep import capacity_sweep
from src.models.base import FeatureMap
from src.models.min_norm import min_norm_fit
from src.models.polynomial import LegendreFeatures
from src.models.random_features import draw_random_relu
from src.theory.threshold import capacity_grid

ModelName = Literal["polynomial", "random_features"]
MODELS: tuple[ModelName, ...] = ("polynomial", "random_features")
MODEL_LABELS: dict[str, str] = {
    "polynomial": "Polynomial (1-D, Legendre)",
    "random_features": "Random ReLU features (20-D)",
}
FeatureFactory = Callable[[np.random.Generator, int], FeatureMap]


def task_for(model: ModelName, settings: Settings) -> Task:
    """The task each model family is studied on."""
    cfg = settings.polynomial.task if model == "polynomial" else settings.random_features.task
    return Task(function=cfg.function, dim=cfg.dim, n_test=cfg.n_test, truth_seed=cfg.truth_seed)


def factory_for(model: ModelName, settings: Settings) -> FeatureFactory:
    """Callable (rng, max_n_params) -> feature map. Random features draw fresh weights per call."""
    if model == "polynomial":
        poly = LegendreFeatures(
            normalize=settings.polynomial.normalize, decay=settings.polynomial.decay
        )

        def make_poly(rng: np.random.Generator, max_n_params: int) -> FeatureMap:
            del rng, max_n_params  # deterministic basis
            return poly

        return make_poly

    rf = settings.random_features

    def make_rf(rng: np.random.Generator, max_n_params: int) -> FeatureMap:
        return draw_random_relu(
            max_n_params,
            rng,
            dim=rf.task.dim,
            weight_scale=rf.weight_scale,
            bias_scale=rf.bias_scale,
        )

    return make_rf


def n_train_for(model: ModelName, settings: Settings) -> int:
    return (
        settings.polynomial.n_train if model == "polynomial" else settings.random_features.n_train
    )


def grid_for(model: ModelName, settings: Settings, n_train: int | None = None) -> np.ndarray:
    """Capacity grid (in n_params) that crosses the threshold for this model family."""
    cfg = settings.polynomial.grid if model == "polynomial" else settings.random_features.grid
    n = n_train if n_train is not None else n_train_for(model, settings)
    return capacity_grid(n, cfg.max_ratio, cfg.n_under, cfg.dense_window, cfg.n_over)


def run_capacity_experiment(
    model: ModelName,
    settings: Settings,
    noise_levels: Sequence[float] | None = None,
    n_seeds: int | None = None,
    n_train: int | None = None,
    n_params_grid: Sequence[int] | IntArray | None = None,
    seed: int | None = None,
    ridge_lambda: float = 0.0,
) -> pd.DataFrame:
    """Capacity sweep for one model family with settings defaults, any of which can be overridden."""
    cfg = settings.polynomial if model == "polynomial" else settings.random_features
    n = n_train if n_train is not None else cfg.n_train
    grid = n_params_grid if n_params_grid is not None else grid_for(model, settings, n)
    return capacity_sweep(
        factory_for(model, settings),
        task_for(model, settings),
        n_train=n,
        noise_levels=noise_levels if noise_levels is not None else settings.noise_levels,
        n_params_grid=grid,
        n_seeds=n_seeds if n_seeds is not None else cfg.n_seeds,
        seed=seed if seed is not None else settings.seed,
        ridge_lambda=ridge_lambda,
    )


def aggregate(
    df: pd.DataFrame,
    value: str = "test_mse",
    by: Sequence[str] = ("model", "n_params", "n_train", "noise"),
) -> pd.DataFrame:
    """Median, quartiles and mean of ``value`` over seeds. Medians are robust to the rare
    near-singular draw that dominates the mean near the threshold."""
    grouped = df.groupby(list(by), sort=True)[value]
    out = grouped.agg(
        median="median",
        q25=lambda s: s.quantile(0.25),
        q75=lambda s: s.quantile(0.75),
        mean="mean",
    )
    return out.reset_index()


@dataclass(frozen=True)
class CurveSummary:
    """Landmarks of one median test-error curve (one model, one noise level)."""

    n_train: int
    classical_min_params: int
    classical_min_error: float
    peak_params: int
    peak_error: float
    second_min_params: int
    second_min_error: float
    final_params: int
    final_error: float

    @property
    def beats_classical(self) -> bool:
        return self.second_min_error < self.classical_min_error

    @property
    def peak_over_classical(self) -> float:
        return self.peak_error / self.classical_min_error

    def to_dict(self) -> dict[str, float | int | bool]:
        out: dict[str, float | int | bool] = dict(asdict(self))
        out["beats_classical"] = self.beats_classical
        out["peak_over_classical"] = self.peak_over_classical
        return out


def summarize_curve(curve: pd.DataFrame, n_train: int, value: str = "median") -> CurveSummary:
    """Landmarks of a curve with columns ``n_params`` and ``value`` (one model, one noise)."""
    c = curve.sort_values("n_params")
    p = c["n_params"].to_numpy()
    v = c[value].to_numpy()
    under = p < n_train
    over = p > n_train
    if not under.any() or not over.any():
        raise ValueError("curve must contain sizes on both sides of the threshold")
    iu = int(np.argmin(np.where(under, v, np.inf)))
    io = int(np.argmin(np.where(over, v, np.inf)))
    ip = int(np.argmax(v))
    return CurveSummary(
        n_train=n_train,
        classical_min_params=int(p[iu]),
        classical_min_error=float(v[iu]),
        peak_params=int(p[ip]),
        peak_error=float(v[ip]),
        second_min_params=int(p[io]),
        second_min_error=float(v[io]),
        final_params=int(p[-1]),
        final_error=float(v[-1]),
    )


@dataclass(frozen=True)
class ExampleFit:
    """One training draw and its fits at several model sizes.

    For the 1-D polynomial track ``x_grid`` is the even test grid, so fits can be drawn as curves.
    For the 20-D random-feature track it is the fixed Gaussian test sample.
    """

    x_train: FloatArray  # (n_train,) or (n_train, dim)
    y_train: FloatArray  # (n_train,) noisy labels
    f_train: FloatArray  # (n_train,) noiseless truth at x_train
    x_grid: FloatArray  # (n_test,) or (n_test, dim)
    f_grid: FloatArray  # (n_test,) truth
    predictions: dict[int, FloatArray]  # n_params -> (n_test,) predictions on x_grid
    train_predictions: dict[int, FloatArray]  # n_params -> (n_train,) predictions on x_train
    train_mse: dict[int, float]
    test_mse: dict[int, float]
    noise: float


def example_fits(
    settings: Settings,
    n_params_list: Sequence[int],
    noise: float | None = None,
    draw: int = 0,
    n_train: int | None = None,
    model: ModelName = "polynomial",
) -> ExampleFit:
    """Fits for one training draw: draw ``draw`` of the capacity sweep with the same seed."""
    sigma = settings.headline_noise if noise is None else noise
    n = n_train if n_train is not None else n_train_for(model, settings)
    task = task_for(model, settings)
    rng = np.random.default_rng(settings.seed).spawn(draw + 1)[draw]
    x = task.sample_x(n, rng)
    f_train = task.f(x)
    y = f_train + sigma * rng.standard_normal(n)
    fmap = factory_for(model, settings)(rng, max(int(v) for v in n_params_list))
    x_grid = task.test_x()
    f_grid = task.f(x_grid)
    preds: dict[int, FloatArray] = {}
    train_preds: dict[int, FloatArray] = {}
    train_mse: dict[int, float] = {}
    test_mse: dict[int, float] = {}
    for p in n_params_list:
        cap = fmap.capacity_for(int(p))
        phi = fmap(x, cap)
        theta = min_norm_fit(phi, y)
        preds[int(p)] = fmap(x_grid, cap) @ theta
        train_preds[int(p)] = phi @ theta
        train_mse[int(p)] = float(np.mean((train_preds[int(p)] - y) ** 2))
        test_mse[int(p)] = float(np.mean((preds[int(p)] - f_grid) ** 2))
    return ExampleFit(
        x, y, f_train, x_grid, f_grid, preds, train_preds, train_mse, test_mse, sigma
    )
