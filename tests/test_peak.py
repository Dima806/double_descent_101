"""Test error peaks exactly at the interpolation threshold, grows with noise, and the same
threshold appears when data grows at fixed capacity."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import Settings
from src.evaluation import aggregate, data_sweep, factory_for, task_for
from src.theory.decomposition import bias_variance
from src.theory.threshold import conditioning_sweep
from tests.conftest import curve_at


def _peak_location(curve: pd.DataFrame) -> tuple[int, int]:
    """(n_params at the maximum, its index) of a median curve."""
    i = int(np.argmax(curve["median"].to_numpy()))
    return int(curve["n_params"].iloc[i]), i


@pytest.mark.parametrize("noise", [0.1, 0.3, 0.5])
@pytest.mark.parametrize("model", ["polynomial", "random_features"])
def test_peak_sits_at_the_threshold(
    model: str,
    noise: float,
    poly_curves: pd.DataFrame,
    rf_curves: pd.DataFrame,
    settings: Settings,
) -> None:
    curves = poly_curves if model == "polynomial" else rf_curves
    n = settings.polynomial.n_train if model == "polynomial" else settings.random_features.n_train
    curve = curve_at(curves, noise)
    p_peak, i_peak = _peak_location(curve)
    i_threshold = int(np.flatnonzero(curve["n_params"].to_numpy() == n)[0])
    assert abs(i_peak - i_threshold) <= 1, f"peak at p={p_peak}, threshold at p={n}"


@pytest.mark.parametrize("model", ["polynomial", "random_features"])
def test_peak_grows_with_noise(
    model: str, poly_curves: pd.DataFrame, rf_curves: pd.DataFrame, settings: Settings
) -> None:
    curves = poly_curves if model == "polynomial" else rf_curves
    n = settings.polynomial.n_train if model == "polynomial" else settings.random_features.n_train
    at_threshold = [
        curve_at(curves, s).set_index("n_params").loc[n, "median"] for s in (0.0, 0.1, 0.3, 0.5)
    ]
    assert all(np.diff(at_threshold) > 0)


def test_noiseless_polynomial_has_no_peak(poly_curves: pd.DataFrame, settings: Settings) -> None:
    """Without noise there is nothing to contort around: the 1-D peak all but vanishes."""
    n = settings.polynomial.n_train
    clean = curve_at(poly_curves, 0.0).set_index("n_params")["median"]
    noisy = curve_at(poly_curves, 0.3).set_index("n_params")["median"]
    assert clean.loc[n] < clean.iloc[0]  # not even above the 1-parameter model
    assert noisy.loc[n] > 1e6 * clean.loc[n]


def test_variance_spikes_at_threshold(settings: Settings) -> None:
    n = settings.random_features.n_train
    dec = bias_variance(
        factory_for("random_features", settings),
        task_for("random_features", settings),
        n_train=n,
        noise=0.3,
        n_params_grid=[n // 2, n, 10 * n],
        n_draws=30,
        seed=settings.seed,
    ).set_index("n_params")
    assert dec.loc[n, "variance"] > 10 * dec.loc[n // 2, "variance"]
    assert dec.loc[n, "variance"] > 10 * dec.loc[10 * n, "variance"]


def test_more_data_can_hurt(settings: Settings) -> None:
    p = settings.data_sweep.n_params
    df = data_sweep(
        factory_for("random_features", settings),
        task_for("random_features", settings),
        n_params=p,
        n_train_grid=settings.data_sweep.n_train_grid,
        noise_levels=[0.3],
        n_seeds=15,
        seed=settings.seed,
    )
    curve = aggregate(df, by=("n_train",)).set_index("n_train")["median"]
    assert int(curve.idxmax()) == p, "peak where n_train == n_params"
    rising = curve.loc[40:p]
    assert all(np.diff(rising.to_numpy()) > 0), "adding data makes it worse from n=40 up to n=p"
    assert curve.loc[p] > 20 * curve.loc[40]


def test_design_matrix_is_nearly_singular_at_threshold(settings: Settings) -> None:
    n = settings.random_features.n_train
    cond = conditioning_sweep(
        factory_for("random_features", settings),
        task_for("random_features", settings),
        n_train=n,
        n_params_grid=[n // 2, n, 2 * n],
        n_seeds=10,
        seed=settings.seed,
    )
    smin = cond.groupby("n_params")["sigma_min"].median()
    assert smin.loc[n] < 0.1 * smin.loc[n // 2]
    assert smin.loc[n] < 0.1 * smin.loc[2 * n]
