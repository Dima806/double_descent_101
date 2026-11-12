"""Optional tiny MLP (skipped without the torch extra)."""

from __future__ import annotations

import numpy as np
import pytest

from src.datasets.synthetic import Task
from src.models.small_net import mlp_n_params, mlp_width_sweep, torch_available


def test_parameter_count() -> None:
    assert mlp_n_params(1, 10) == 31
    assert mlp_n_params(20, 5) == 111


def test_torch_flag_is_boolean() -> None:
    assert isinstance(torch_available(), bool)


def test_trained_net_fits_its_training_data() -> None:
    pytest.importorskip("torch")
    df = mlp_width_sweep(
        Task("two_tone", n_test=200),
        n_train=12,
        widths=[1, 64],
        noise=0.1,
        epochs=600,
        learning_rate=0.01,
        n_seeds=1,
        seed=0,
    )
    small, wide = df.sort_values("capacity")["train_mse"].to_numpy()
    assert wide < small
    assert wide < 1e-2
    assert np.all(np.isfinite(df["test_mse"]))
