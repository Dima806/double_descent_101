"""Synthetic regression tasks with a known true function plus Gaussian noise.

Double descent is a statement about test error against the truth, so the truth must be known.

Two kinds of task:
- 1-D (``dim == 1``): x ~ U[-1, 1], test inputs on an even grid. Used for polynomials, where the
  fitted curve can be drawn.
- d-D (``dim > 1``): x ~ N(0, I_d), test inputs a fixed Gaussian sample. Used for random features;
  high dimension is where the second descent beats the classical sweet spot.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from src.arrays import FloatArray


def _sine(x: FloatArray) -> FloatArray:
    return np.sin(np.pi * x)


def _two_tone(x: FloatArray) -> FloatArray:
    return np.sin(np.pi * x) + 0.5 * np.cos(2.0 * np.pi * x)


def _wave(x: FloatArray) -> FloatArray:
    return np.sin(2.0 * np.pi * x)


def _bump(x: FloatArray) -> FloatArray:
    return np.exp(-4.0 * x**2) * np.cos(3.0 * x)


FUNCTIONS_1D: dict[str, Callable[[FloatArray], FloatArray]] = {
    "sine": _sine,
    "two_tone": _two_tone,
    "wave": _wave,
    "bump": _bump,
}
FUNCTIONS_ND: tuple[str, ...] = ("tanh_ridge",)
FUNCTION_FORMULAS: dict[str, str] = {
    "sine": "sin(pi x)",
    "two_tone": "sin(pi x) + 0.5 cos(2 pi x)",
    "wave": "sin(2 pi x)",
    "bump": "exp(-4 x^2) cos(3 x)",
    "tanh_ridge": "tanh(2 <beta, x>), beta a fixed random unit vector",
}


def ridge_direction(dim: int, truth_seed: int) -> FloatArray:
    """Fixed random unit vector beta that defines the d-D target. Shape (dim,)."""
    beta = np.random.default_rng(truth_seed).standard_normal(dim)
    return beta / np.linalg.norm(beta)


@dataclass(frozen=True)
class Task:
    """A known true function f and the input distribution it is sampled on."""

    function: str
    dim: int = 1
    n_test: int = 2000
    truth_seed: int = 0

    def __post_init__(self) -> None:
        if self.dim == 1 and self.function not in FUNCTIONS_1D:
            known = ", ".join(sorted(FUNCTIONS_1D))
            raise ValueError(f"Unknown 1-D function {self.function!r}; choose from {known}")
        if self.dim > 1 and self.function not in FUNCTIONS_ND:
            known = ", ".join(FUNCTIONS_ND)
            raise ValueError(f"Unknown d-D function {self.function!r}; choose from {known}")
        if self.dim < 1:
            raise ValueError("dim must be >= 1")

    @property
    def formula(self) -> str:
        return FUNCTION_FORMULAS[self.function]

    def f(self, x: FloatArray) -> FloatArray:
        """True function. x: (n,) for dim == 1, (n, dim) otherwise. Returns (n,)."""
        if self.dim == 1:
            return FUNCTIONS_1D[self.function](np.asarray(x, dtype=np.float64))
        beta = ridge_direction(self.dim, self.truth_seed)
        return np.tanh(2.0 * (np.asarray(x, dtype=np.float64) @ beta))

    def sample_x(self, n: int, rng: np.random.Generator) -> FloatArray:
        """Training inputs: U[-1, 1] (shape (n,)) or N(0, I_d) (shape (n, dim))."""
        if self.dim == 1:
            return rng.uniform(-1.0, 1.0, size=n)
        return rng.standard_normal((n, self.dim))

    def test_x(self) -> FloatArray:
        """Fixed test inputs: even grid on [-1, 1] or a Gaussian sample seeded by truth_seed."""
        if self.dim == 1:
            return np.linspace(-1.0, 1.0, self.n_test)
        rng = np.random.default_rng(self.truth_seed + 1_000_003)
        return rng.standard_normal((self.n_test, self.dim))

    def signal_variance(self) -> float:
        """E[f(x)^2] on the test inputs: the error of always predicting 0 (the null predictor)."""
        return float(np.mean(self.f(self.test_x()) ** 2))


@dataclass(frozen=True)
class Dataset:
    """One training draw plus the fixed, noiseless test set."""

    x_train: FloatArray  # (n_train,) or (n_train, dim)
    y_train: FloatArray  # (n_train,) noisy targets
    f_train: FloatArray  # (n_train,) noiseless truth at x_train
    x_test: FloatArray  # (n_test,) or (n_test, dim)
    f_test: FloatArray  # (n_test,) noiseless truth: test error is measured against this
    noise: float

    @property
    def n_train(self) -> int:
        return int(self.x_train.shape[0])


def make_dataset(task: Task, n_train: int, noise: float, rng: np.random.Generator) -> Dataset:
    """Draw y = f(x) + noise * N(0, 1) on ``n_train`` inputs and attach the fixed test set."""
    x_train = task.sample_x(n_train, rng)
    f_train = task.f(x_train)
    y_train = f_train + noise * rng.standard_normal(n_train)
    x_test = task.test_x()
    return Dataset(
        x_train=x_train,
        y_train=y_train,
        f_train=f_train,
        x_test=x_test,
        f_test=task.f(x_test),
        noise=noise,
    )
