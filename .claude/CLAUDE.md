# Project: double_descent_101

## Identity
Educational "101" portfolio repo: double descent built from scratch, CPU only.
Deliverables (all present): 6 executed notebooks, `src/` library, Streamlit app, 18 PNG figures with
JSON data, and the article draft `overfitting_is_only_half_the_curve.md` (Medium, ~2,750 words).

Thesis: the bias-variance U is only HALF the curve. Sweep capacity past the interpolation threshold
(n_params == n_train): test error SPIKES at the threshold, then DESCENDS AGAIN with train error = 0,
and in high dimension beats the classical sweet spot. Min-norm least squares on polynomial / random
features shows it in seconds; a tiny trained MLP confirms the shape.

Arc (one notebook each): classical U → interpolation peak → second descent (showpiece) →
benign overfitting → more data can hurt → what it means for you.

Sibling repos (cross-link, don't duplicate): `Dima806/regularization_101` (ridge flattens the peak),
`Dima806/curse_of_dimensionality_101` (high-dim geometry where benign overfitting lives),
`Dima806/cv_arena` (estimating test error, hardest near the threshold). README lists more.

Full PRD: `.llm/PRD_double_descent_101.md` (gitignored, local only), so this file must stand alone.

## Stack and hardware
- 2 CPU / 8 GB Codespace, no GPU. Python >=3.11 (system is 3.14), deps via `uv` only, `uv.lock` committed.
- Runtime: numpy, scipy, scikit-learn, pandas, matplotlib, plotly, streamlit>=1.50, pydantic,
  pydantic-settings, pyyaml. Optional extra `torch` (CPU index) for the MLP only.
- Dev: pytest, pytest-cov, ruff, ty, jupyter, ipykernel, nbconvert, nbclient. Licence Apache-2.0.

## Commands
```
make setup       # install uv if missing, uv sync --all-extras, register kernel
make lint        # ruff format + ruff check --fix + ty check src/   (MUTATES files)
make lint-check  # same checks, read-only
make test        # pytest, 76 tests, ~30 s
make test-cov    # pytest --cov=src
make notebooks   # execute notebooks/0*.ipynb IN PLACE (~7.5 min total, each < 4 min), rewrites outputs/
make run         # streamlit on :8501 (APP_PORT=...), headless
make ci          # uv sync --locked + lint-check + test  (must stay green)
```
Single test: `uv run pytest tests/test_peak.py -k polynomial`. After editing pyproject: `uv lock`.

## The two experimental tracks (settled by experiment; don't change casually)
| Track | Task (`config/settings.yaml`) | Model | n | Grid | Role |
|---|---|---|---|---|---|
| `polynomial` | 1-D, x~U[-1,1], `two_tone` = sin(pi x)+0.5cos(2 pi x) | Legendre, scaled sqrt(2k+1)/(1+k) (`decay: 1`) | 15 | p 1..90 | intuition + drawable fits |
| `random_features` | 20-D, x~N(0,I), `tanh_ridge` = tanh(2<beta,x>) | ReLU max(0,w.x+b), w~N(0,I/d), b~U(-1,1) | 100 | p 1..2000 | the "beats sweet spot" result |
Noise levels [0, 0.1, 0.3, 0.5], headline 0.3, seed 2026.

Why two tracks (found the hard way):
- 1-D random features do NOT show double descent (no peak at p=n, error grows with width): in fixed low
  dimension interpolating noise is at best "tempered". High dimension is where benign overfitting lives.
- 1-D polynomials peak exactly at p=n but the peak is huge (median ~3e9 at σ=0.3) because LS at random
  points is exponentially ill-conditioned; second descent reaches 0.16 vs sweet spot 0.12 (does NOT beat).
- `decay` only changes the OVERparameterized min-norm interpolant (below p=n LS is basis-independent):
  decay 0 plateaus near the null predictor (0.54), decay 1 reaches 0.16. This is the "which interpolant" lesson.
- Polynomial errors are heavy-tailed (rare draw with near-duplicate x): MEANS explode from p≈6. Use
  medians for polynomial curves; the mean-based bias-variance decomposition uses random features only.

Headline numbers (σ=0.3, deterministic): poly sweet spot p=5 0.124, peak p=15 3.2e9, p=90 0.159;
RF sweet spot p=32 0.417, peak p=100 71.4, p=2000 0.243 (42% below); RF beats sweet spot at all σ
(51/51/42/27%). Data sweep p=100: best n=30 0.48, n=100 53, n=400 0.21. Ridge per-point λ: 1e-2 removes
the peak, λ=1 best 0.160. MLP: width 1 0.12, peak width 5 (111 params) 0.96, width 136 0.24.

## Layout
```
config/settings.yaml        every number (tracks, grids, seeds, draws, data sweep, heatmap, ridge, mlp, paths, dpi)
src/config.py               pydantic-settings; YAML loaded by load_settings(); env DD_<SECTION>__<KEY> overrides; get_settings() cached
src/arrays.py               FloatArray / IntArray aliases
src/datasets/synthetic.py   Task(function, dim, n_test, truth_seed): f(), sample_x(), test_x(), signal_variance(); make_dataset()
src/models/base.py          FeatureMap protocol (name property, __call__(x, capacity), n_params, capacity_for); maps are NESTED
src/models/polynomial.py    LegendreFeatures(normalize, decay); monomial_features (only to show ill-conditioning)
src/models/random_features.py  RandomReLUFeatures + draw_random_relu(max_width, rng, dim, weight_scale, bias_scale)
src/models/min_norm.py      min_norm_fit (SVD, eps*max(n,p) cutoff), ridge_fit(phi, y, lam), predict
src/models/small_net.py     torch_available, mlp_n_params, train_mlp, mlp_width_sweep (lazy torch import)
src/theory/threshold.py     interpolation_threshold, param_ratio, regime, capacity_grid, conditioning_sweep (sigma_min)
src/theory/decomposition.py bias_variance (mean-based, exact identity), signal_noise_split (linearity split of the fit)
src/evaluation/capacity_sweep.py  capacity_sweep(factory, task, n_train, noise_levels, grid, n_seeds, seed, ridge_lambda=0)
src/evaluation/data_sweep.py      data_sweep (fixed p, nested n), error_surface (p x n grid)
src/evaluation/comparison.py      task_for, factory_for, grid_for, run_capacity_experiment, aggregate, summarize_curve
                                  (CurveSummary), example_fits (ExampleFit, both tracks), MODEL_LABELS
src/visualisation.py        palette + apply_style, capacity_axes, plot_curve/band/fit/decomposition/heatmap, annotate_point,
                            format_error, noise_colors, save_figure; plotly_capacity_curve / plotly_fit for the app
src/reporting.py            to_jsonable, save_json/load_json, publish_figure (PNG + JSON in one call)
app/streamlit_app.py        4 tabs: capacity sweep, the fit (polynomial), noise dial, more data can hurt
tests/                      conftest (session sweeps), test_{u_curve,peak,second_descent,min_norm} + datasets, threshold,
                            config, reporting, small_net, app (AppTest)
notebooks/                  01..06, committed WITH outputs
outputs/figures/            18 PNGs (cap: 19 total). outputs/data/  one JSON per figure + NN_summary.json per notebook
overfitting_is_only_half_the_curve.md   article draft
```
Dependency direction: datasets / models / theory → evaluation → visualisation / reporting → notebooks / app.

## Contracts
- FeatureFactory = `(rng, max_n_params) -> FeatureMap`. Random features draw fresh weights per call; polynomial ignores rng.
- Draw order per seed is fixed everywhere: `x = sample_x`, `eps = standard_normal(n)`, then `factory(rng, ...)`.
  So `capacity_sweep`, `signal_noise_split`, `conditioning_sweep` and `example_fits(draw=k)` see identical draws
  (tests rely on this). Seeds: `np.random.default_rng(seed).spawn(n_seeds)`.
- Sweep rows: `model, capacity, n_params, n_train, noise, seed, train_mse, test_mse, coef_norm`.
  `aggregate()` → `median, q25, q75, mean`; `summarize_curve()` → classical min / peak / second min / final.
- Every notebook figure goes through `publish_figure(fig, stem, title, caption, data)`: writes
  `outputs/figures/<stem>.png` and `outputs/data/<stem>.json` (title, caption, plotted numbers). Each notebook ends with
  `save_json(..., "NN_summary")` (claim, settings, results, figures, takeaways). Stems: `NN_slug`.

## Core definitions
- `n_params` counts every fitted coefficient: poly degree d → d+1; RF width N → N (bias inside the ReLU).
- Threshold p == n. Test error = MSE vs NOISELESS f on the fixed test set (2,000 points). Train error on noisy y.
- Min-norm = pseudo-inverse = ridge λ→0+. Ridge λ is per training point: `ridge_fit(phi, y, lam * n)`.
- Bias² / variance over independent draws (fresh x, noise, features); bias² + variance == expected MSE exactly.

## Numerical pitfalls
1. Monomials are useless past degree ~15-20; use Legendre on [-1, 1].
2. Never raise the SVD cutoff, add jitter or ridge to "tame" the peak: the peak IS the result.
3. No hidden regularization: no sklearn `Ridge()` default alpha, no `LinearRegression(fit_intercept=True)`, no MLP weight decay.
4. Log-y and medians for curves; the mean near p=n is dominated by rare near-singular draws.
5. Grids must contain p == n exactly and be dense near it (`capacity_grid`).
6. 3 of 200 polynomial fits at p ≈ n cannot interpolate in float64 (near-duplicate x); tests require median
   train MSE ≈ 0 and ≥ 98% of fits interpolating, not 100%.
7. 2 cores: vectorize over noise levels (one SVD, many y columns); no nested parallelism.

## Testing
76 tests, ~30 s. Required claims: U interior minimum (both tracks), peak at p=n ±1 grid step (both tracks, σ>0),
peak grows with noise, noiseless polynomial has no peak, variance spike, sigma_min collapse, more data hurts,
second descent (final < 1% of peak, falling trend), train ≈ 0 past threshold, RF beats sweet spot at every σ,
polynomial does NOT beat (keeps the article honest), min-norm == pinv/lstsq/OLS/ridgeless. Never weaken an
assertion to get green: fix the experiment or report.

## Notebooks
- Committed with outputs, kernelspec `python3`. Edit cells directly (they are the source), then `make notebooks`.
- Thin: settings → src calls → plot → `publish_figure`; every figure has a caption + JSON. Markdown narrative quotes
  numbers that must match the outputs: after changing config, re-check every number in the markdown AND the article.
- Runtimes on 2 CPUs: 01 ~55 s, 02 ~35 s, 03 ~110 s, 04 ~40 s, 05 ~55 s, 06 ~155 s (MLP + 6 ridge sweeps).
- Figure budget: max 19 PNGs total (now 18). Adding a figure means removing or merging one.

## Figure style (`src/visualisation.py`, validated palette)
- Categorical: polynomial blue `#2a78d6`, random features orange `#eb6834`, MLP aqua `#1baf7a` (aqua < 3:1 contrast,
  so always legend/label it). Ordered quantities (noise, λ): one blue ramp, light → dark, max 4 lines per plot.
- Log-y, dashed threshold line labeled "p = n = …", regime washes with labels at the bottom (`capacity_axes` adds
  bottom headroom so labels sit below data), IQR bands, text in ink colours only, labels on translucent boxes.
- Fonts: titles bold (DejaVu has no semibold). Check every PNG by eye after changes: no overlaps, nothing cut off.

## Article (`overfitting_is_only_half_the_curve.md`)
Follows the author's latest article structure (diffusion_101): `# X 101: title`, `## _italic first-person subtitle_`,
3-paragraph intro ending "every number below comes from a notebook or a test", `[IMAGE: name.png]` + blank line +
`Source: author`, claim-style `##` headings, one debugging story, "What I would take away from this" bullets, closing
paragraph with repo URL + `make setup` / `make notebooks` + sibling links, numbered `## References`.
Style: plain ASCII (no em dashes, Greek letters or math symbols: write sigma, lambda, "times"), first person, no
AI-cliche phrasing, every number traceable to `outputs/data/*.json`.

## Reader-facing writing (notebooks, README, article)
Full prose (caveman mode never applies). Honest: classical U first; "can"/"often" unless asserted; the lesson is
regime-dependent, not "bigger always wins"; regularization still matters.

## Open items
- `.python-version` not pinned (system Python 3.14 works with current wheels).
- `.claude/skills/caveman.md` referenced by the PRD does not exist (gitignored dir).

## Token rules
- Caveman mode ON for chat with the user: terse, no filler, code and numbers first. /compact at notebook boundaries.
- Never cat executed notebooks (huge JSON): read cell sources via python/json or `jupyter nbconvert --to script`.
