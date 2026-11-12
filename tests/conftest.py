"""Shared, session-cached sweeps. Smaller than the notebook runs, same qualitative setup."""

from __future__ import annotations

import pandas as pd
import pytest

from src.config import Settings, get_settings
from src.evaluation import aggregate, run_capacity_experiment

NOISES = [0.0, 0.1, 0.3, 0.5]


@pytest.fixture(scope="session")
def settings() -> Settings:
    return get_settings()


@pytest.fixture(scope="session")
def poly_sweep(settings: Settings) -> pd.DataFrame:
    return run_capacity_experiment("polynomial", settings, noise_levels=NOISES, n_seeds=80)


@pytest.fixture(scope="session")
def rf_sweep(settings: Settings) -> pd.DataFrame:
    return run_capacity_experiment("random_features", settings, noise_levels=NOISES, n_seeds=12)


@pytest.fixture(scope="session")
def poly_curves(poly_sweep: pd.DataFrame) -> pd.DataFrame:
    return aggregate(poly_sweep)


@pytest.fixture(scope="session")
def rf_curves(rf_sweep: pd.DataFrame) -> pd.DataFrame:
    return aggregate(rf_sweep)


def curve_at(curves: pd.DataFrame, noise: float) -> pd.DataFrame:
    """One median curve (sorted by n_params) at a given noise level."""
    return curves[curves["noise"] == noise].sort_values("n_params").reset_index(drop=True)
