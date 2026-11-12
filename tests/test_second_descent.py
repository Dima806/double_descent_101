"""Past the threshold: error descends again with zero training error; in 20-D it beats the
classical sweet spot."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import Settings
from src.evaluation import factory_for, run_capacity_experiment, summarize_curve, task_for
from src.theory.decomposition import signal_noise_split
from tests.conftest import curve_at


def _n(model: str, settings: Settings) -> int:
    return (
        settings.polynomial.n_train if model == "polynomial" else settings.random_features.n_train
    )


def _curves(model: str, poly: pd.DataFrame, rf: pd.DataFrame) -> pd.DataFrame:
    return poly if model == "polynomial" else rf


@pytest.mark.parametrize("noise", [0.1, 0.3, 0.5])
@pytest.mark.parametrize("model", ["polynomial", "random_features"])
def test_error_descends_again_past_threshold(
    model: str,
    noise: float,
    poly_curves: pd.DataFrame,
    rf_curves: pd.DataFrame,
    settings: Settings,
) -> None:
    n = _n(model, settings)
    curve = curve_at(_curves(model, poly_curves, rf_curves), noise)
    summary = summarize_curve(curve, n)
    assert summary.final_error < 0.01 * summary.peak_error
    beyond = curve[curve["n_params"] > 1.2 * n]
    ranks = np.argsort(np.argsort(beyond["median"].to_numpy()))
    assert np.corrcoef(beyond["n_params"], ranks)[0, 1] < -0.8, "error keeps falling with capacity"


@pytest.mark.parametrize("model", ["polynomial", "random_features"])
def test_training_error_is_zero_past_threshold(
    model: str, poly_sweep: pd.DataFrame, rf_sweep: pd.DataFrame, settings: Settings
) -> None:
    sweep = poly_sweep if model == "polynomial" else rf_sweep
    over = sweep[(sweep["n_params"] >= _n(model, settings)) & (sweep["noise"] > 0)]
    rel = over["train_mse"] / over["noise"] ** 2
    assert rel.median() < 1e-12
    # a handful of polynomial draws have two near-identical x values; exact interpolation would
    # need coefficients beyond float64, so the SVD cutoff stops just short
    assert (rel < 1e-8).mean() > 0.98


@pytest.mark.parametrize("noise", [0.0, 0.1, 0.3, 0.5])
def test_random_features_beat_the_classical_sweet_spot(
    rf_curves: pd.DataFrame, settings: Settings, noise: float
) -> None:
    summary = summarize_curve(curve_at(rf_curves, noise), _n("random_features", settings))
    assert summary.beats_classical
    assert summary.second_min_params > 5 * summary.n_train  # the win is far past the threshold


def test_polynomial_second_descent_stays_above_classical_minimum(
    poly_curves: pd.DataFrame, settings: Settings
) -> None:
    """The 1-D contrast the notebooks describe: descent yes, beating the sweet spot no."""
    summary = summarize_curve(curve_at(poly_curves, 0.3), _n("polynomial", settings))
    assert not summary.beats_classical
    assert summary.second_min_error < 5 * summary.classical_min_error  # close, but above


def test_noise_footprint_shrinks_with_capacity(settings: Settings) -> None:
    n = settings.random_features.n_train
    grid = [n, 2 * n, 20 * n]
    split = signal_noise_split(
        factory_for("random_features", settings),
        task_for("random_features", settings),
        n_train=n,
        noise=0.3,
        n_params_grid=grid,
        n_seeds=8,
        seed=settings.seed,
    )
    med = split.groupby("n_params").median()
    assert med.loc[20 * n, "noise_energy"] < 0.05 * med.loc[n, "noise_energy"]
    assert med.loc[20 * n, "noise_energy"] < med.loc[2 * n, "noise_energy"]
    assert (split["train_mse"] < 1e-15).all()  # the noise is memorised exactly ...
    sweep = run_capacity_experiment(
        "random_features", settings, noise_levels=[0.3], n_seeds=8, n_params_grid=grid
    )
    merged = split.merge(sweep, on=["n_params", "seed"])
    np.testing.assert_allclose(merged["total"], merged["test_mse"], rtol=1e-8)
