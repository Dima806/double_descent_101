.DEFAULT_GOAL := help

.PHONY: help setup sync lint lint-check format check typecheck test test-cov \
        notebooks run lab clean reset ci dev

LINT_DIRS   := src/ tests/ app/
KERNEL_NAME := double-descent-101
APP_PORT    ?= 8501
LAB_PORT    ?= 8888
NB_TIMEOUT  ?= 240

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

setup: ## First-time setup (uv, all extras, Jupyter kernel)
	@command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
	uv sync --all-extras
	uv run python -m ipykernel install --user --name $(KERNEL_NAME) --display-name "$(KERNEL_NAME)"
	@printf "\n✅ Ready. Run 'make test'.\n"

sync: ## Sync deps (all extras)
	uv sync --all-extras

lint: format check typecheck ## All linters (auto-fixes files)

lint-check: ## All linters, read-only (used by CI)
	uv run ruff format --check $(LINT_DIRS)
	uv run ruff check $(LINT_DIRS)
	uv run ty check src/

format: ## ruff format
	uv run ruff format $(LINT_DIRS)

check: ## ruff check --fix
	uv run ruff check --fix $(LINT_DIRS)

typecheck: ## ty check
	uv run ty check src/

test: ## pytest
	uv run pytest

test-cov: ## pytest with coverage
	uv run pytest --cov=src --cov-report=term-missing

notebooks: ## Execute all notebooks in place (outputs are committed)
	@for nb in notebooks/0*.ipynb; do \
		[ -e "$$nb" ] || { echo "No notebooks found."; exit 0; }; \
		echo "▶ Executing $$nb ..."; \
		start=$$(date +%s); \
		uv run jupyter nbconvert --to notebook --execute --inplace \
			--ExecutePreprocessor.timeout=$(NB_TIMEOUT) \
			--ExecutePreprocessor.kernel_name=python3 "$$nb" || exit 1; \
		echo "  ⏱  $$(( $$(date +%s) - start ))s (budget: 240s per notebook)"; \
	done; \
	printf "\n✅ All notebooks executed.\n"

run: ## Streamlit app
	uv run streamlit run app/streamlit_app.py --server.port $(APP_PORT) --server.headless true

lab: ## JupyterLab
	uv run jupyter lab --no-browser --port $(LAB_PORT)

clean: ## Remove caches and stray nbconvert copies
	find . -type d -name __pycache__ -not -path "./.venv/*" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ipynb_checkpoints -not -path "./.venv/*" -exec rm -rf {} + 2>/dev/null || true
	find notebooks -name "*.nbconvert.ipynb" -delete 2>/dev/null || true
	rm -rf htmlcov .coverage .pytest_cache .ruff_cache
	@printf "🧹 Cleaned.\n"

reset: clean ## Full reset (also removes .venv)
	rm -rf .venv

ci: ## CI: locked sync, read-only lint, tests
	uv sync --all-extras --locked
	$(MAKE) lint-check test

dev: lint test ## Fast loop
