"""Settings load, validate and accept environment overrides."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from src.config import PROJECT_ROOT, get_settings, load_settings, resolve_path


def test_default_settings_load() -> None:
    s = get_settings()
    assert s.polynomial.n_train == 15
    assert s.random_features.task.dim == 20
    assert s.headline_noise in s.noise_levels
    assert s.figures_path == PROJECT_ROOT / "outputs" / "figures"
    assert s.data_path.is_absolute()


def test_environment_overrides_yaml(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DD_RANDOM_FEATURES__N_SEEDS", "3")
    monkeypatch.setenv("DD_SEED", "11")
    s = load_settings()
    assert s.random_features.n_seeds == 3
    assert s.seed == 11
    assert s.random_features.n_train == get_settings().random_features.n_train


def test_unknown_keys_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    text = (PROJECT_ROOT / "config" / "settings.yaml").read_text() + "\nmystery_knob: 1\n"
    bad.write_text(text)
    with pytest.raises(ValidationError):
        load_settings(bad)


def test_resolve_path() -> None:
    assert resolve_path(Path("a/b")) == PROJECT_ROOT / "a" / "b"
    assert resolve_path(Path("/tmp/x")) == Path("/tmp/x")
