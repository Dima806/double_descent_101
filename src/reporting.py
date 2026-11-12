"""JSON outputs: every number and caption behind a figure is saved next to it as data."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT, Settings, get_settings


def to_jsonable(obj: Any) -> Any:
    """Recursively convert numpy / pandas / pathlib / dataclass objects to JSON-safe values.
    Non-finite floats become None (JSON has no inf/nan)."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, pd.DataFrame):
        return to_jsonable(obj.to_dict(orient="list"))
    if isinstance(obj, pd.Series):
        return to_jsonable(obj.tolist())
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, float | np.floating):
        val = float(obj)
        return val if math.isfinite(val) else None
    if isinstance(obj, Path):
        return str(obj)
    if is_dataclass(obj) and not isinstance(obj, type):
        return to_jsonable(asdict(obj))
    if hasattr(obj, "model_dump"):
        return to_jsonable(obj.model_dump(mode="json"))
    return obj


def save_json(payload: dict[str, Any], stem: str, settings: Settings | None = None) -> Path:
    """Write ``payload`` to ``<data_dir>/<stem>.json`` (pretty-printed) and return the path."""
    s = settings or get_settings()
    out_dir = s.data_path
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stem}.json"
    path.write_text(json.dumps(to_jsonable(payload), indent=2) + "\n", encoding="utf-8")
    return path


def load_json(stem: str, settings: Settings | None = None) -> dict[str, Any]:
    """Read ``<data_dir>/<stem>.json``."""
    s = settings or get_settings()
    return json.loads((s.data_path / f"{stem}.json").read_text(encoding="utf-8"))


def publish_figure(
    fig: Any,
    stem: str,
    title: str,
    caption: str,
    data: dict[str, Any],
    settings: Settings | None = None,
) -> dict[str, str]:
    """Save a figure as PNG and everything behind it (title, caption, plotted numbers) as JSON.

    Writes ``<figures_dir>/<stem>.png`` and ``<data_dir>/<stem>.json``; returns both paths.
    """
    from src.visualisation import save_figure  # local import keeps reporting matplotlib-free

    s = settings or get_settings()
    png = save_figure(fig, stem, s)
    payload = {
        "figure": stem,
        "png": str(png.relative_to(PROJECT_ROOT))
        if png.is_relative_to(PROJECT_ROOT)
        else str(png),
        "title": title,
        "caption": caption,
        "data": data,
    }
    js = save_json(payload, stem, s)
    return {"png": str(png), "json": str(js)}
