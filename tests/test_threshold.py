"""Threshold bookkeeping: parameter counts, regimes and capacity grids."""

from __future__ import annotations

import numpy as np
import pytest

from src.models.polynomial import LegendreFeatures, monomial_features
from src.models.random_features import draw_random_relu
from src.theory.threshold import capacity_grid, interpolation_threshold, param_ratio, regime


def test_threshold_and_regimes() -> None:
    assert interpolation_threshold(15) == 15
    assert regime(14, 15) == "underparameterized"
    assert regime(15, 15) == "threshold"
    assert regime(16, 15) == "overparameterized"
    assert param_ratio(30, 15) == 2.0
    with pytest.raises(ValueError, match="n_train"):
        interpolation_threshold(0)


@pytest.mark.parametrize(("n", "ratio", "n_under"), [(15, 6.0, None), (100, 20.0, 25)])
def test_grid_contains_threshold_and_is_dense_near_it(
    n: int, ratio: float, n_under: int | None
) -> None:
    grid = capacity_grid(n, ratio, n_under=n_under, dense_window=0.15)
    assert n in grid
    assert grid[0] == 1
    assert grid[-1] == round(ratio * n)
    assert np.all(np.diff(grid) > 0)
    near = grid[(grid >= 0.85 * n) & (grid <= 1.15 * n)]
    assert np.max(np.diff(near)) <= max(1, n // 40)
    with pytest.raises(ValueError, match="max_ratio"):
        capacity_grid(n, 1.0)


def test_polynomial_parameter_count_and_nesting() -> None:
    fm = LegendreFeatures(decay=1.0)
    x = np.linspace(-1, 1, 7)
    assert fm(x, 4).shape == (7, 5)
    assert fm.n_params(4) == 5 and fm.capacity_for(5) == 4
    np.testing.assert_allclose(fm(x, 9)[:, :5], fm(x, 4))  # nested
    with pytest.raises(ValueError, match="degree"):
        fm(x, -1)


def test_legendre_is_far_better_conditioned_than_monomials() -> None:
    x = np.random.default_rng(0).uniform(-1, 1, 200)
    cond_leg = np.linalg.cond(LegendreFeatures(decay=0.0)(x, 25))
    cond_mono = np.linalg.cond(monomial_features(x, 25))
    assert cond_mono > 1e4 * cond_leg


def test_random_features_parameter_count_and_nesting() -> None:
    fm = draw_random_relu(50, np.random.default_rng(1), dim=3)
    x = np.random.default_rng(2).standard_normal((6, 3))
    assert fm(x, 20).shape == (6, 20)
    assert fm.n_params(20) == 20 and fm.capacity_for(20) == 20
    np.testing.assert_allclose(fm(x, 50)[:, :20], fm(x, 20))
    assert np.all(fm(x, 50) >= 0)
    with pytest.raises(ValueError, match="width"):
        fm(x, 51)
    one_d = draw_random_relu(5, np.random.default_rng(3))
    assert one_d(np.linspace(-1, 1, 4), 5).shape == (4, 5)
