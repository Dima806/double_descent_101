"""Typed settings loaded from ``config/settings.yaml``.

Every experiment number lives in the YAML file. Environment variables prefixed ``DD_`` override
it, with ``__`` for nesting (e.g. ``DD_RANDOM_FEATURES__N_SEEDS=10``). Paths are resolved from the
repository root, so notebooks (cwd ``notebooks/``) and the app (``app/``) find the same files.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "settings.yaml"


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TaskConfig(_Section):
    function: str
    dim: int = Field(1, ge=1)
    n_test: int = Field(2000, ge=10)
    truth_seed: int = 0


class GridConfig(_Section):
    max_ratio: float = Field(gt=1.0)
    n_under: int | None = None
    dense_window: float = Field(0.15, gt=0.0, lt=1.0)
    n_over: int = Field(20, ge=2)


class PolynomialConfig(_Section):
    task: TaskConfig
    n_train: int = Field(ge=2)
    decay: float = Field(ge=0.0)
    normalize: bool = True
    grid: GridConfig
    n_seeds: int = Field(ge=1)
    decomposition_draws: int = Field(ge=2)


class RandomFeaturesConfig(_Section):
    task: TaskConfig
    n_train: int = Field(ge=2)
    weight_scale: float = Field(gt=0.0)
    bias_scale: float = Field(ge=0.0)
    grid: GridConfig
    n_seeds: int = Field(ge=1)
    decomposition_draws: int = Field(ge=2)


class DataSweepConfig(_Section):
    n_params: int = Field(ge=1)
    n_train_grid: list[int]
    n_seeds: int = Field(ge=1)


class HeatmapConfig(_Section):
    n_params_grid: list[int]
    n_train_grid: list[int]
    n_seeds: int = Field(ge=1)


class RidgeConfig(_Section):
    lambdas: list[float]
    n_seeds: int = Field(ge=1)


class MLPConfig(_Section):
    n_train: int = Field(ge=2)
    widths: list[int]
    epochs: int = Field(ge=1)
    learning_rate: float = Field(gt=0.0)
    n_seeds: int = Field(ge=1)
    noise: float = Field(ge=0.0)
    threads: int = Field(2, ge=1)


class PathsConfig(_Section):
    figures_dir: Path = Path("outputs/figures")
    data_dir: Path = Path("outputs/data")


class PlotConfig(_Section):
    dpi: int = Field(150, ge=50)


class Settings(BaseSettings):
    """Root settings object. Build it with ``get_settings()`` or ``load_settings(path)``."""

    model_config = SettingsConfigDict(
        env_prefix="DD_", env_nested_delimiter="__", extra="forbid", frozen=True
    )

    seed: int
    noise_levels: list[float]
    headline_noise: float
    polynomial: PolynomialConfig
    random_features: RandomFeaturesConfig
    data_sweep: DataSweepConfig
    heatmap: HeatmapConfig
    ridge: RidgeConfig
    mlp: MLPConfig
    paths: PathsConfig = PathsConfig()
    plots: PlotConfig = PlotConfig()

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Environment beats the YAML values passed in by ``load_settings``.
        return (env_settings, init_settings)

    @property
    def figures_path(self) -> Path:
        return resolve_path(self.paths.figures_dir)

    @property
    def data_path(self) -> Path:
        return resolve_path(self.paths.data_dir)


def resolve_path(path: Path) -> Path:
    """Absolute path; relative paths are taken from the repository root."""
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_settings(path: Path | str | None = None) -> Settings:
    """Read and validate a settings YAML file (default: ``config/settings.yaml``)."""
    yaml_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    data: dict[str, Any] = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    return Settings(**data)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached default settings."""
    return load_settings()
