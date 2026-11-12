"""Synthetic tasks with a known true function."""

from src.datasets.synthetic import (
    FUNCTION_FORMULAS,
    FUNCTIONS_1D,
    FUNCTIONS_ND,
    Dataset,
    Task,
    make_dataset,
    ridge_direction,
)

__all__ = [
    "FUNCTIONS_1D",
    "FUNCTIONS_ND",
    "FUNCTION_FORMULAS",
    "Dataset",
    "Task",
    "make_dataset",
    "ridge_direction",
]
