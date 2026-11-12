"""All plotting: matplotlib for notebooks and saved PNGs, plotly for the Streamlit app.

Conventions (shared by every figure):
- log-y for test error; a dashed vertical line at the interpolation threshold (p = n);
  under/overparameterized regions lightly washed and labeled;
- median over seeds as the line, interquartile range as a 15% band;
- colour follows identity: polynomial = blue, random features = orange, MLP = aqua;
  ordered quantities (noise level, ridge strength) use one blue ramp, light -> dark;
- text is always ink (never series colour); legends for 2+ series.
Functions draw on a given Axes (or return a plotly Figure) and never call ``plt.show()``.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib.axes import Axes
from matplotlib.collections import QuadMesh
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.figure import Figure

from src.arrays import FloatArray
from src.config import Settings, get_settings

# --- palette (validated: categorical slots 1-3 all-pairs, ordinal blue ramp) ---
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MODEL_COLORS = {"polynomial": SERIES[0], "random_features": SERIES[1], "mlp": SERIES[2]}
BLUE_RAMP = {
    100: "#cde2fb",
    150: "#b7d3f6",
    200: "#9ec5f4",
    250: "#86b6ef",
    300: "#6da7ec",
    350: "#5598e7",
    400: "#3987e5",
    450: "#2a78d6",
    500: "#256abf",
    550: "#1c5cab",
    600: "#184f95",
    650: "#104281",
    700: "#0d366b",
}
UNDER_WASH = "#f1f0ec"
OVER_WASH = "#eaf2fc"
BIAS_COLOR = SERIES[0]
VARIANCE_COLOR = SERIES[1]
TRAIN_COLOR = MUTED
POINT_COLOR = SERIES[1]


def apply_style() -> None:
    """Set the shared matplotlib style (readable at article width)."""
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "sans-serif",
            "font.size": 12,
            "axes.titlesize": 13.5,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.titlepad": 10,
            "axes.labelsize": 12.5,
            "axes.labelcolor": INK_2,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 1.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "grid.linestyle": "-",
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "legend.frameon": True,
            "legend.framealpha": 0.92,
            "legend.edgecolor": GRID,
            "lines.linewidth": 2.0,
            "lines.solid_capstyle": "round",
            "text.color": INK,
            "figure.titlesize": 15,
            "figure.titleweight": "bold",
        }
    )


def new_figure(
    ncols: int = 1, nrows: int = 1, width: float | None = None, height: float | None = None
) -> tuple[Figure, list[Axes]]:
    """Figure with constrained layout and a flat list of axes. Default ~5.6 x 4.2 in per panel."""
    apply_style()
    w = width if width is not None else 5.6 * ncols
    h = height if height is not None else 4.2 * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(w, h), layout="constrained", squeeze=False)
    return fig, list(axes.ravel())


def format_error(value: float) -> str:
    """Compact error label: 0.12, 3.4, 1.1e+04."""
    return f"{value:.2f}" if abs(value) < 100 else f"{value:.1e}"


def noise_colors(levels: Sequence[float]) -> dict[float, str]:
    """Ordered blue ramp for noise levels, lightest = least noise."""
    ordered = sorted(float(lv) for lv in levels)
    if len(ordered) <= 4:
        steps = [250, 400, 550, 700][4 - len(ordered) :] if len(ordered) > 1 else [450]
        return {lv: BLUE_RAMP[st] for lv, st in zip(ordered, steps, strict=True)}
    return dict(zip(ordered, ordered_colors(len(ordered)), strict=True))


def ordered_colors(n: int) -> list[str]:
    """``n`` blue-ramp steps, light -> dark (for ordered series such as ridge strength)."""
    keys = sorted(BLUE_RAMP)[3:]  # start at step 250 so the lightest line clears the surface
    picks = np.linspace(0, len(keys) - 1, n).round().astype(int)
    return [BLUE_RAMP[keys[i]] for i in picks]


# --- annotations -------------------------------------------------------------------------


def mark_threshold(ax: Axes, x: float, label: str | None = "interpolation threshold") -> None:
    """Dashed vertical reference line at the threshold, labeled near the top."""
    ax.axvline(x, color=INK, lw=1.3, ls=(0, (5, 3)), zorder=3)
    if label:
        ax.annotate(
            label,
            xy=(x, 1.0),
            xycoords=("data", "axes fraction"),
            xytext=(4, -4),
            textcoords="offset points",
            ha="left",
            va="top",
            fontsize=10.5,
            color=INK,
            bbox={"boxstyle": "round,pad=0.2", "fc": SURFACE, "ec": "none", "alpha": 0.8},
            zorder=6,
        )


def shade_regimes(
    ax: Axes, x_threshold: float, labels: bool = True, flipped: bool = False
) -> None:
    """Wash the under- and overparameterized sides; label them at the bottom of the axes.

    ``flipped=True`` is for a training-set-size axis, where small n (left) is overparameterized.
    """
    lo, hi = ax.get_xlim()
    left, right = (OVER_WASH, UNDER_WASH) if flipped else (UNDER_WASH, OVER_WASH)
    ax.axvspan(lo, x_threshold, color=left, zorder=0, lw=0)
    ax.axvspan(x_threshold, hi, color=right, zorder=0, lw=0)
    ax.set_xlim(lo, hi)
    names = ["underparameterized", "overparameterized"]
    if flipped:
        names.reverse()
    if labels:
        for text, dx, ha in ((names[0], -6, "right"), (names[1], 6, "left")):
            ax.annotate(
                text,
                xy=(x_threshold, 0.0),
                xycoords=("data", "axes fraction"),
                xytext=(dx, 6),
                textcoords="offset points",
                ha=ha,
                va="bottom",
                fontsize=10,
                color=INK_2,
                bbox={"boxstyle": "round,pad=0.2", "fc": SURFACE, "ec": "none", "alpha": 0.8},
                zorder=6,
            )


def plot_band(
    ax: Axes,
    x: FloatArray,
    median: FloatArray,
    q25: FloatArray | None,
    q75: FloatArray | None,
    color: str,
    label: str | None,
    ls: str = "-",
    lw: float = 2.0,
    zorder: int = 4,
) -> None:
    """Median line with an optional interquartile band."""
    if q25 is not None and q75 is not None:
        ax.fill_between(x, q25, q75, color=color, alpha=0.15, lw=0, zorder=zorder - 1)
    ax.plot(x, median, color=color, ls=ls, lw=lw, label=label, zorder=zorder)


def plot_curve(
    ax: Axes,
    agg: pd.DataFrame,
    color: str,
    label: str | None,
    x: str = "n_params",
    band: bool = True,
    ls: str = "-",
    lw: float = 2.0,
) -> None:
    """Draw an ``aggregate`` frame (columns x, median, q25, q75) as median + IQR band."""
    a = agg.sort_values(x)
    plot_band(
        ax,
        a[x].to_numpy(dtype=float),
        a["median"].to_numpy(),
        a["q25"].to_numpy() if band else None,
        a["q75"].to_numpy() if band else None,
        color,
        label,
        ls=ls,
        lw=lw,
    )


def capacity_axes(
    ax: Axes,
    n_train: int,
    xlabel: str = "number of parameters p",
    ylabel: str = "test MSE vs true f (log scale)",
    logx: bool = False,
    regimes: bool = True,
    threshold_label: str | None = None,
    flipped: bool = False,
) -> None:
    """Common dressing for an error-vs-capacity panel (call after plotting the data).

    The log y-axis gets extra room at the bottom so the regime labels sit below the data."""
    ax.set_yscale("log")
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo / (hi / lo) ** 0.12, hi)
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if regimes:
        shade_regimes(ax, n_train, flipped=flipped)
    mark_threshold(
        ax, n_train, f"p = n = {n_train}" if threshold_label is None else threshold_label
    )


def annotate_point(
    ax: Axes,
    x: float,
    y: float,
    text: str,
    dx: float = 10,
    dy: float = 10,
    ha: str = "left",
    color: str = INK,
) -> None:
    """Small ringed dot plus a short label with a leader line."""
    ax.plot([x], [y], "o", ms=7, color=color, mec=SURFACE, mew=2, zorder=6)
    ax.annotate(
        text,
        xy=(x, y),
        xytext=(dx, dy),
        textcoords="offset points",
        ha=ha,
        va="center",
        fontsize=10.5,
        color=INK,
        bbox={"boxstyle": "round,pad=0.25", "fc": SURFACE, "ec": "none", "alpha": 0.85},
        arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.8, "shrinkA": 0, "shrinkB": 4},
        zorder=6,
    )


# --- specific panels ----------------------------------------------------------------------


def plot_fit(
    ax: Axes,
    x_grid: FloatArray,
    f_grid: FloatArray,
    x_train: FloatArray,
    y_train: FloatArray,
    pred: FloatArray,
    color: str = SERIES[0],
    label: str = "fitted model",
    ylim: tuple[float, float] = (-2.6, 2.6),
    title: str | None = None,
) -> None:
    """Truth, noisy training points and one fitted curve; annotates a fit that leaves the chart."""
    ax.plot(x_grid, f_grid, color=INK, lw=1.6, label="true function", zorder=3)
    ax.plot(
        x_grid,
        np.clip(pred, ylim[0] - 10, ylim[1] + 10),
        color=color,
        lw=2.2,
        label=label,
        zorder=4,
    )
    ax.plot(
        x_train,
        y_train,
        "o",
        ms=7.5,
        color=POINT_COLOR,
        mec=SURFACE,
        mew=1.8,
        label="noisy training points",
        zorder=5,
    )
    ax.set_ylim(*ylim)
    ax.set_xlim(float(np.min(x_grid)), float(np.max(x_grid)))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    peak = float(np.max(np.abs(pred)))
    if peak > ylim[1]:
        ax.annotate(
            f"fit leaves the chart: max |ŷ| ≈ {peak:.0e}",
            xy=(0.5, 0.03),
            xycoords="axes fraction",
            ha="center",
            va="bottom",
            fontsize=10.5,
            color=INK,
            bbox={"boxstyle": "round,pad=0.3", "fc": SURFACE, "ec": GRID},
            zorder=7,
        )
    if title:
        ax.set_title(title)


def plot_decomposition(ax: Axes, dec: pd.DataFrame, n_train: int, show_total: bool = True) -> None:
    """Bias^2, variance and their sum (expected test MSE) against model size."""
    d = dec.sort_values("n_params")
    x = d["n_params"].to_numpy(dtype=float)
    ax.plot(x, d["bias2"], color=BIAS_COLOR, label="bias²")
    ax.plot(x, d["variance"], color=VARIANCE_COLOR, label="variance")
    if show_total:
        ax.plot(
            x,
            d["expected_mse"],
            color=INK,
            lw=1.6,
            ls=(0, (1, 1.6)),
            label="bias² + variance (test MSE)",
        )
    del n_train


def error_cmap() -> LinearSegmentedColormap:
    """Sequential blue colormap for error magnitude (light = low error)."""
    keys = sorted(BLUE_RAMP)
    return LinearSegmentedColormap.from_list("dd_blue", [BLUE_RAMP[k] for k in keys])


def plot_heatmap(
    ax: Axes,
    table: pd.DataFrame,
    vmin: float | None = None,
    vmax: float | None = None,
) -> QuadMesh:
    """Heatmap of a pivot table (index n_train, columns n_params) on log colour scale."""
    ps = table.columns.to_numpy(dtype=float)
    ns = table.index.to_numpy(dtype=float)
    values = table.to_numpy()

    def edges(v: FloatArray) -> FloatArray:
        mids = np.sqrt(v[:-1] * v[1:])
        return np.concatenate([[v[0] ** 2 / mids[0]], mids, [v[-1] ** 2 / mids[-1]]])

    mesh = ax.pcolormesh(
        edges(ps),
        edges(ns),
        values,
        cmap=error_cmap(),
        norm=LogNorm(vmin=vmin or float(np.nanmin(values)), vmax=vmax or float(np.nanmax(values))),
        shading="flat",
        rasterized=True,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    lim = (max(ps.min(), ns.min()), min(ps.max(), ns.max()))
    ax.plot(lim, lim, color=SURFACE, lw=1.4, ls=(0, (5, 3)))
    ax.grid(False)
    return mesh


def save_figure(fig: Figure, stem: str, settings: Settings | None = None) -> Path:
    """Save ``fig`` as ``<figures_dir>/<stem>.png`` at the configured dpi; return the path."""
    s = settings or get_settings()
    out_dir = s.figures_path
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stem}.png"
    fig.savefig(path, dpi=s.plots.dpi, facecolor=SURFACE)
    return path


# --- plotly (Streamlit app) ---------------------------------------------------------------

_PLOTLY_LAYOUT = {
    "plot_bgcolor": SURFACE,
    "paper_bgcolor": SURFACE,
    "font": {"family": "system-ui, -apple-system, Segoe UI, sans-serif", "size": 13, "color": INK},
    "margin": {"l": 60, "r": 20, "t": 50, "b": 55},
    "legend": {"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
    "hovermode": "x unified",
}


def _hex_to_rgba(color: str, alpha: float) -> str:
    c = color.lstrip("#")
    r, g, b = (int(c[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def _plotly_axes(fig: go.Figure, xtitle: str, ytitle: str, logy: bool, logx: bool = False) -> None:
    fig.update_xaxes(
        title=xtitle,
        type="log" if logx else "linear",
        gridcolor=GRID,
        linecolor=AXIS,
        zeroline=False,
    )
    fig.update_yaxes(
        title=ytitle,
        type="log" if logy else "linear",
        gridcolor=GRID,
        linecolor=AXIS,
        zeroline=False,
    )


def plotly_capacity_curve(
    curves: dict[str, pd.DataFrame],
    colors: dict[str, str],
    n_train: int,
    current: int | None = None,
    title: str = "",
    logx: bool = False,
    x: str = "n_params",
    xtitle: str = "number of parameters p",
    threshold_text: str | None = None,
) -> go.Figure:
    """Median + IQR curves (one per key) with threshold line and an optional current-size marker."""
    fig = go.Figure()
    for name, agg in curves.items():
        a = agg.sort_values(x)
        col = colors[name]
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([a[x], a[x][::-1]]),
                y=np.concatenate([a["q75"], a["q25"][::-1]]),
                fill="toself",
                fillcolor=_hex_to_rgba(col, 0.15),
                line={"width": 0},
                hoverinfo="skip",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=a[x],
                y=a["median"],
                mode="lines",
                name=name,
                line={"color": col, "width": 2.5},
                hovertemplate="%{y:.3g}",
            )
        )
        if current is not None:
            row = a.iloc[(a[x] - current).abs().argmin()]
            fig.add_trace(
                go.Scatter(
                    x=[row[x]],
                    y=[row["median"]],
                    mode="markers",
                    marker={"size": 13, "color": col, "line": {"color": SURFACE, "width": 2}},
                    name=f"{name}: selected",
                    showlegend=False,
                    hovertemplate=f"{x}=%{{x}}<br>median test MSE=%{{y:.3g}}<extra></extra>",
                )
            )
    fig.add_vline(
        x=n_train,
        line={"color": INK, "dash": "dash", "width": 1.5},
        annotation_text=threshold_text or f"threshold p = n = {n_train}",
        annotation_position="top right",
    )
    fig.update_layout(title=title, **_PLOTLY_LAYOUT)
    _plotly_axes(fig, xtitle, "median test MSE vs true f (log)", logy=True, logx=logx)
    return fig


def plotly_fit(
    x_grid: FloatArray,
    f_grid: FloatArray,
    x_train: FloatArray,
    y_train: FloatArray,
    pred: FloatArray,
    title: str = "",
    ylim: tuple[float, float] = (-2.6, 2.6),
) -> go.Figure:
    """Truth, training points and the fitted curve (clipped to ``ylim``)."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_grid, y=f_grid, mode="lines", name="true function", line={"color": INK, "width": 2}
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_grid,
            y=np.clip(pred, ylim[0] - 5, ylim[1] + 5),
            mode="lines",
            name="fitted model",
            line={"color": SERIES[0], "width": 2.5},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_train,
            y=y_train,
            mode="markers",
            name="noisy training points",
            marker={"size": 10, "color": POINT_COLOR, "line": {"color": SURFACE, "width": 2}},
        )
    )
    fig.update_layout(title=title, **_PLOTLY_LAYOUT)
    fig.update_layout(hovermode="closest")
    _plotly_axes(fig, "x", "y", logy=False)
    fig.update_yaxes(range=list(ylim))
    return fig
