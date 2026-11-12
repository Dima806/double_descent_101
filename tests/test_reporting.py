"""Figures are saved as PNG and their numbers and captions as JSON."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import PathsConfig, Settings
from src.reporting import load_json, publish_figure, save_json, to_jsonable
from src.visualisation import (
    capacity_axes,
    new_figure,
    noise_colors,
    ordered_colors,
    plot_band,
    plot_fit,
)


def _tmp_settings(settings: Settings, tmp_path: Path) -> Settings:
    paths = PathsConfig(figures_dir=tmp_path / "fig", data_dir=tmp_path / "data")
    return settings.model_copy(update={"paths": paths})


def test_to_jsonable_handles_numpy_pandas_and_non_finite() -> None:
    out = to_jsonable(
        {
            "a": np.arange(3),
            "b": np.float64(np.inf),
            "c": np.bool_(True),
            "d": pd.DataFrame({"x": [1, 2]}),
            "e": (np.int64(4), float("nan")),
            "f": Path("p/q"),
        }
    )
    assert out == {
        "a": [0, 1, 2],
        "b": None,
        "c": True,
        "d": {"x": [1, 2]},
        "e": [4, None],
        "f": "p/q",
    }
    json.dumps(out)


def test_save_and_load_json(settings: Settings, tmp_path: Path) -> None:
    s = _tmp_settings(settings, tmp_path)
    path = save_json({"value": np.float64(1.5)}, "x", s)
    assert path.exists()
    assert load_json("x", s) == {"value": 1.5}


def test_publish_figure_writes_png_and_json(settings: Settings, tmp_path: Path) -> None:
    s = _tmp_settings(settings, tmp_path)
    fig, (ax,) = new_figure()
    x = np.arange(1.0, 31.0)
    y = 1 + (x - 15) ** 2
    plot_band(ax, x, y, 0.8 * y, 1.2 * y, noise_colors([0.3])[0.3], "curve")
    capacity_axes(ax, 15)
    out = publish_figure(fig, "demo", "Demo", "A caption.", {"x": x, "y": y}, s)
    assert Path(out["png"]).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    payload = json.loads(Path(out["json"]).read_text())
    assert payload["caption"] == "A caption." and payload["data"]["x"][0] == 1.0


def test_plot_fit_flags_a_fit_that_leaves_the_chart() -> None:
    fig, (ax,) = new_figure()
    xg = np.linspace(-1, 1, 50)
    plot_fit(ax, xg, np.sin(xg), xg[::10], np.sin(xg[::10]), 1e4 * xg)
    texts = [t.get_text() for t in ax.texts]
    assert any("leaves the chart" in t for t in texts)


def test_colour_helpers() -> None:
    cols = noise_colors([0.5, 0.0, 0.3, 0.1])
    assert list(cols) == [0.0, 0.1, 0.3, 0.5]
    assert len(set(cols.values())) == 4
    assert len(set(ordered_colors(6))) == 6
