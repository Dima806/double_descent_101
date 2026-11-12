"""Sweeps across capacity and data size, plus the harness that runs and summarises them."""

from src.evaluation.capacity_sweep import capacity_sweep
from src.evaluation.comparison import (
    CurveSummary,
    ExampleFit,
    aggregate,
    example_fits,
    factory_for,
    grid_for,
    run_capacity_experiment,
    summarize_curve,
    task_for,
)
from src.evaluation.data_sweep import data_sweep, error_surface

__all__ = [
    "CurveSummary",
    "ExampleFit",
    "aggregate",
    "capacity_sweep",
    "data_sweep",
    "error_surface",
    "example_fits",
    "factory_for",
    "grid_for",
    "run_capacity_experiment",
    "summarize_curve",
    "task_for",
]
