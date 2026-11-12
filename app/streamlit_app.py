"""Double Descent 101 — interactive companion to the notebooks.

Run with ``make run`` (http://localhost:8501). All computation comes from ``src``; this file is UI.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.config import get_settings
from src.evaluation import (
    ExampleFit,
    aggregate,
    data_sweep,
    example_fits,
    factory_for,
    grid_for,
    run_capacity_experiment,
    summarize_curve,
    task_for,
)
from src.evaluation.comparison import MODEL_LABELS, ModelName
from src.theory.threshold import regime
from src.visualisation import MODEL_COLORS, noise_colors, plotly_capacity_curve, plotly_fit

SETTINGS = get_settings()
NOISE_CHOICES = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8]
APP_SEEDS: dict[str, int] = {"polynomial": 60, "random_features": 12}
MODEL_BY_LABEL: dict[str, ModelName] = {
    "Polynomial (1-D)": "polynomial",
    "Random features (20-D)": "random_features",
}


@st.cache_data(show_spinner="Sweeping capacity across the threshold ...")
def capacity_curves(model: ModelName, n_train: int, n_seeds: int) -> pd.DataFrame:
    """Median/IQR test error for every noise choice (one sweep; noise levels share draws)."""
    df = run_capacity_experiment(
        model, SETTINGS, noise_levels=NOISE_CHOICES, n_seeds=n_seeds, n_train=n_train
    )
    test = aggregate(df)
    train = aggregate(df, value="train_mse")[["n_params", "noise", "median"]]
    return test.merge(train.rename(columns={"median": "train_median"}), on=["n_params", "noise"])


@st.cache_data(show_spinner="Growing the training set ...")
def data_curve(n_params: int, noise: float, n_seeds: int) -> pd.DataFrame:
    df = data_sweep(
        factory_for("random_features", SETTINGS),
        task_for("random_features", SETTINGS),
        n_params=n_params,
        n_train_grid=sorted({*SETTINGS.data_sweep.n_train_grid, n_params}),
        noise_levels=[noise],
        n_seeds=n_seeds,
        seed=SETTINGS.seed,
    )
    return aggregate(df, by=("n_train", "noise"))


@st.cache_data
def polynomial_fit(n_params: int, noise: float, n_train: int, draw: int) -> ExampleFit:
    return example_fits(SETTINGS, [n_params], noise=noise, draw=draw, n_train=n_train)


def describe(n_params: int, n_train: int) -> str:
    return {
        "underparameterized": "Underparameterized: cannot fit every training point.",
        "threshold": "At the threshold: exactly one interpolant, forced through every noisy point.",
        "overparameterized": "Overparameterized: many interpolants; min-norm picks the smoothest.",
    }[regime(n_params, n_train)]


st.set_page_config(page_title="Double Descent 101", layout="wide")
st.title("Double Descent 101")
st.markdown(
    "Test error against model size is not just a U. It **spikes** where the number of parameters "
    "equals the number of training points, then **falls again**. Move the sliders and watch."
)

with st.sidebar:
    st.header("Setup")
    model_label = st.radio("Model family", list(MODEL_BY_LABEL), index=0)
    model = MODEL_BY_LABEL[model_label]
    noise = st.select_slider("Label noise σ", options=NOISE_CHOICES, value=SETTINGS.headline_noise)
    if model == "polynomial":
        n_train = st.slider("Training points n", 10, 25, SETTINGS.polynomial.n_train)
    else:
        n_train = SETTINGS.random_features.n_train
        st.caption(f"Random features use n = {n_train} points in 20 dimensions.")
    st.caption(
        f"Medians over {APP_SEEDS[model]} training draws; test error is measured against the "
        "noiseless truth."
    )

curves = capacity_curves(model, n_train, APP_SEEDS[model])
grid = [int(p) for p in grid_for(model, SETTINGS, n_train)]
color = MODEL_COLORS[model]
tab_sweep, tab_fit, tab_noise, tab_data = st.tabs(
    ["1 · Capacity sweep", "2 · The fit", "3 · Noise dial", "4 · More data can hurt"]
)

with tab_sweep:
    p = st.select_slider("Number of parameters p", options=grid, value=grid[len(grid) // 3])
    curve = curves[curves["noise"] == noise]
    row = curve.set_index("n_params").loc[p]
    summary = summarize_curve(curve, n_train)
    cols = st.columns(4)
    cols[0].metric("p / n", f"{p / n_train:.2f}")
    cols[1].metric("Median test MSE", f"{row['median']:.3g}")
    cols[2].metric("Median train MSE", f"{row['train_median']:.2g}")
    cols[3].metric(
        "Classical sweet spot",
        f"{summary.classical_min_error:.3g} at p={summary.classical_min_params}",
    )
    st.plotly_chart(
        plotly_capacity_curve(
            {MODEL_LABELS[model]: curve},
            {MODEL_LABELS[model]: color},
            n_train,
            current=p,
            title=f"{MODEL_LABELS[model]}, σ = {noise}",
            logx=model == "random_features",
        ),
        width="stretch",
    )
    st.info(describe(p, n_train))
    verdict = "beats" if summary.beats_classical else "does not beat"
    st.caption(
        f"Peak: {summary.peak_error:.3g} at p = {summary.peak_params}. Best past the threshold: "
        f"{summary.second_min_error:.3g} at p = {summary.second_min_params}, which {verdict} "
        "the classical sweet spot."
    )

with tab_fit:
    st.caption("Fits are drawn for the 1-D polynomial track, where the curve can be seen.")
    n_poly = n_train if model == "polynomial" else SETTINGS.polynomial.n_train
    poly_grid = [int(v) for v in grid_for("polynomial", SETTINGS, n_poly)]
    c1, c2 = st.columns([3, 1])
    p_fit = c1.select_slider(
        "Polynomial parameters p (degree + 1)", options=poly_grid, value=min(5, n_poly - 1)
    )
    draw = c2.number_input("Training draw", min_value=0, max_value=19, value=0, step=1)
    fit = polynomial_fit(p_fit, noise, n_poly, int(draw))
    st.plotly_chart(
        plotly_fit(
            fit.x_grid,
            fit.f_grid,
            fit.x_train,
            fit.y_train,
            fit.predictions[p_fit],
            title=f"p = {p_fit}, n = {n_poly}: train MSE {fit.train_mse[p_fit]:.2g}, "
            f"test MSE {fit.test_mse[p_fit]:.3g}",
        ),
        width="stretch",
    )
    st.info(describe(p_fit, n_poly))

with tab_noise:
    shown = [0.0, 0.1, 0.3, 0.8]
    cmap = noise_colors(shown)
    labelled = {f"σ = {s}": curves[curves["noise"] == s] for s in shown}
    st.plotly_chart(
        plotly_capacity_curve(
            labelled,
            {f"σ = {s}": cmap[s] for s in shown},
            n_train,
            title="The peak is made of noise: more noise, taller spike",
            logx=model == "random_features",
        ),
        width="stretch",
    )
    at_threshold = curves[curves["n_params"] == n_train].set_index("noise")["median"]
    st.dataframe(
        at_threshold.rename("median test MSE at p = n").to_frame().T,
        width="stretch",
    )

with tab_data:
    st.caption("Random features with a fixed width; only the training-set size changes.")
    width = st.select_slider("Fixed number of parameters p", options=[50, 100, 200], value=100)
    dc = data_curve(width, noise, APP_SEEDS["random_features"])
    st.plotly_chart(
        plotly_capacity_curve(
            {"test error": dc},
            {"test error": MODEL_COLORS["random_features"]},
            width,
            title=f"Fixed p = {width}: adding data walks into the peak at n = p",
            logx=True,
            x="n_train",
            xtitle="training-set size n",
            threshold_text=f"threshold n = p = {width}",
        ),
        width="stretch",
    )
    by_n = dc.set_index("n_train")["median"]
    half = int(by_n.index[abs(by_n.index - width / 2).argmin()])
    st.metric(
        f"Test MSE going from n = {half} to n = {width}",
        f"{by_n.loc[width]:.3g}",
        delta=f"{by_n.loc[width] - by_n.loc[half]:+.3g} (worse)",
        delta_color="inverse",
    )
