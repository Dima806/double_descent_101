"""The Streamlit app renders without errors for both model families."""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"


def test_app_renders_both_model_families() -> None:
    at = AppTest.from_file(str(APP), default_timeout=180).run()
    assert not at.exception
    assert len(at.tabs) == 4
    at.sidebar.radio[0].set_value("Random features (20-D)").run()
    assert not at.exception
    at.select_slider[0].set_value(100).run()  # the threshold itself
    assert not at.exception
