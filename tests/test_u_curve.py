"""Underparameterized side: the classical U, with bias falling and variance rising."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import Settings
from src.evaluation import factory_for, task_for
from src.theory.decomposition import bias_variance
from tests.conftest import curve_at


def _underparameterized(curve: pd.DataFrame, n_train: int) -> pd.DataFrame:
    return curve[curve["n_params"] < n_train]


@pytest.mark.parametrize("noise", [0.1, 0.3, 0.5])
def test_polynomial_test_error_is_u_shaped(
    poly_curves: pd.DataFrame, settings: Settings, noise: float
) -> None:
    under = _underparameterized(curve_at(poly_curves, noise), settings.polynomial.n_train)
    err = under["median"].to_numpy()
    i_min = int(np.argmin(err))
    assert 0 < i_min < len(err) - 1, "minimum must be interior: falls, then rises"
    assert err[0] > 2 * err[i_min]  # too simple underfits
    assert err[-1] > 2 * err[i_min]  # too complex (still below threshold) overfits


@pytest.mark.parametrize("noise", [0.1, 0.3, 0.5])
def test_random_features_test_error_is_u_shaped(
    rf_curves: pd.DataFrame, settings: Settings, noise: float
) -> None:
    under = _underparameterized(curve_at(rf_curves, noise), settings.random_features.n_train)
    err = under["median"].to_numpy()
    i_min = int(np.argmin(err))
    assert 0 < i_min < len(err) - 1
    assert err[0] > 1.3 * err[i_min]
    assert err[-1] > 2 * err[i_min]


def test_training_error_falls_with_capacity(poly_sweep: pd.DataFrame, settings: Settings) -> None:
    d = poly_sweep[
        (poly_sweep["noise"] == 0.3) & (poly_sweep["n_params"] < settings.polynomial.n_train)
    ]
    train = d.groupby("n_params")["train_mse"].median().to_numpy()
    assert np.all(np.diff(train) <= 1e-12), "least squares on nested features never fits worse"


@pytest.fixture(scope="module")
def rf_decomposition(settings: Settings) -> pd.DataFrame:
    """Mean-based decomposition on the 20-D random-feature task (polynomial errors are too
    heavy-tailed for means: a rare draw with two clustered inputs dominates them)."""
    return bias_variance(
        factory_for("random_features", settings),
        task_for("random_features", settings),
        n_train=settings.random_features.n_train,
        noise=0.3,
        n_params_grid=[1, 5, 10, 20, 30, 50, 70, 90],
        n_draws=40,
        seed=settings.seed,
    )


def test_bias_falls_and_variance_rises(rf_decomposition: pd.DataFrame) -> None:
    d = rf_decomposition.sort_values("n_params")
    assert d["bias2"].iloc[-1] < 0.3 * d["bias2"].iloc[0]
    assert d["variance"].iloc[-1] > 10 * d["variance"].iloc[0]
    assert np.all(np.diff(d["variance"].to_numpy()) > 0)
    total = d["expected_mse"].to_numpy()
    assert 0 < int(np.argmin(total)) < len(total) - 1  # the classical U, in expectation


def test_decomposition_identity_is_exact(rf_decomposition: pd.DataFrame) -> None:
    d = rf_decomposition
    np.testing.assert_allclose(d["bias2"] + d["variance"], d["expected_mse"], rtol=1e-8)
