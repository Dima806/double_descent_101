"""The min-norm solver is the pseudo-inverse solution, interpolates past the threshold, and is
the ridgeless limit of ridge."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from src.models.min_norm import default_rtol, min_norm_fit, predict, ridge_fit


def _problem(n: int, p: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, p)), rng.standard_normal(n)


@pytest.mark.parametrize(("n", "p"), [(30, 10), (20, 20), (15, 60)])
def test_matches_pseudo_inverse_and_lstsq(n: int, p: int) -> None:
    phi, y = _problem(n, p)
    theta = min_norm_fit(phi, y)
    np.testing.assert_allclose(theta, np.linalg.pinv(phi) @ y, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(
        theta, np.linalg.lstsq(phi, y, rcond=None)[0], rtol=1e-8, atol=1e-10
    )


def test_interpolates_when_overparameterized() -> None:
    phi, y = _problem(15, 60)
    np.testing.assert_allclose(predict(phi, min_norm_fit(phi, y)), y, atol=1e-10)


def test_solution_has_smallest_norm_among_interpolants() -> None:
    phi, y = _problem(10, 40, seed=1)
    theta = min_norm_fit(phi, y)
    _, s, vt = np.linalg.svd(phi)
    null_basis = vt[s.size :].T  # (p, p - n): directions phi cannot see
    assert np.linalg.norm(null_basis.T @ theta) < 1e-10  # theta lies in the row space
    rng = np.random.default_rng(2)
    for _ in range(20):
        other = theta + null_basis @ rng.standard_normal(null_basis.shape[1])
        np.testing.assert_allclose(phi @ other, y, atol=1e-9)  # also interpolates ...
        assert np.linalg.norm(other) > np.linalg.norm(theta)  # ... but has a larger norm


def test_equals_ordinary_least_squares_when_underparameterized() -> None:
    phi, y = _problem(50, 8, seed=3)
    ols = np.linalg.solve(phi.T @ phi, phi.T @ y)
    np.testing.assert_allclose(min_norm_fit(phi, y), ols, rtol=1e-9)


def test_is_the_ridgeless_limit_of_ridge() -> None:
    phi, y = _problem(12, 50, seed=4)
    theta = min_norm_fit(phi, y)
    np.testing.assert_allclose(ridge_fit(phi, y, 1e-10), theta, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(ridge_fit(phi, y, 0.0), theta)
    # ridge with a real penalty shrinks the norm and stops interpolating
    shrunk = ridge_fit(phi, y, 1.0)
    assert np.linalg.norm(shrunk) < np.linalg.norm(theta)
    assert np.max(np.abs(phi @ shrunk - y)) > 1e-3


def test_ridge_rejects_negative_penalty() -> None:
    phi, y = _problem(5, 5)
    with pytest.raises(ValueError, match="non-negative"):
        ridge_fit(phi, y, -1.0)


def test_multiple_targets_solved_column_by_column() -> None:
    phi, _ = _problem(10, 25, seed=5)
    ys = np.random.default_rng(6).standard_normal((10, 3))
    thetas = min_norm_fit(phi, ys)
    assert thetas.shape == (25, 3)
    for j in range(3):
        np.testing.assert_allclose(thetas[:, j], min_norm_fit(phi, ys[:, j]), atol=1e-12)
    np.testing.assert_allclose(
        ridge_fit(phi, ys, 0.5)[:, 1], ridge_fit(phi, ys[:, 1], 0.5), atol=1e-12
    )


def test_agrees_with_sklearn_without_intercept() -> None:
    phi, y = _problem(12, 30, seed=7)
    sk = LinearRegression(fit_intercept=False).fit(phi, y)
    np.testing.assert_allclose(min_norm_fit(phi, y), sk.coef_, rtol=1e-6, atol=1e-8)


def test_large_cutoff_acts_as_hidden_regularization() -> None:
    """Truncating small singular values changes the answer: never use it to tame the peak."""
    rng = np.random.default_rng(8)
    phi = rng.standard_normal((20, 20)) @ np.diag(np.geomspace(1, 1e-6, 20))
    y = rng.standard_normal(20)
    exact = min_norm_fit(phi, y)
    truncated = min_norm_fit(phi, y, rtol=1e-3)
    assert np.linalg.norm(truncated) < 1e-2 * np.linalg.norm(exact)
    assert default_rtol(20, 20) < 1e-13
