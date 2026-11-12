"""Synthetic tasks: determinism, shapes, validation."""

from __future__ import annotations

import numpy as np
import pytest

from src.datasets.synthetic import Task, make_dataset, ridge_direction


def test_same_seed_same_data() -> None:
    task = Task("two_tone")
    a = make_dataset(task, 15, 0.3, np.random.default_rng(5))
    b = make_dataset(task, 15, 0.3, np.random.default_rng(5))
    np.testing.assert_array_equal(a.x_train, b.x_train)
    np.testing.assert_array_equal(a.y_train, b.y_train)
    assert a.n_train == 15


def test_one_dimensional_task() -> None:
    task = Task("two_tone", n_test=101)
    ds = make_dataset(task, 20, 0.0, np.random.default_rng(0))
    assert ds.x_train.shape == (20,)
    assert np.all(np.abs(ds.x_train) <= 1)
    np.testing.assert_allclose(ds.y_train, ds.f_train)  # zero noise
    np.testing.assert_allclose(ds.x_test, np.linspace(-1, 1, 101))
    assert 0.5 < task.signal_variance() < 0.8
    assert "cos" in task.formula


def test_high_dimensional_task() -> None:
    task = Task("tanh_ridge", dim=20, n_test=500)
    ds = make_dataset(task, 30, 0.3, np.random.default_rng(0))
    assert ds.x_train.shape == (30, 20)
    assert ds.x_test.shape == (500, 20)
    np.testing.assert_array_equal(task.test_x(), task.test_x())  # fixed test set
    assert np.all(np.abs(ds.f_train) < 1)
    assert np.std(ds.y_train - ds.f_train) == pytest.approx(0.3, rel=0.35)
    beta = ridge_direction(20, 0)
    assert np.linalg.norm(beta) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("function", "dim"), [("nope", 1), ("two_tone", 5), ("tanh_ridge", 1), ("sine", 0)]
)
def test_invalid_tasks_rejected(function: str, dim: int) -> None:
    with pytest.raises(ValueError):
        Task(function, dim=dim)


@pytest.mark.parametrize("function", ["sine", "two_tone", "wave", "bump"])
def test_all_one_dimensional_functions_are_bounded(function: str) -> None:
    f = Task(function).f(np.linspace(-1, 1, 50))
    assert np.all(np.isfinite(f)) and np.max(np.abs(f)) <= 1.6
