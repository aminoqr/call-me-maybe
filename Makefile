# Makefile for call-me-maybe
.PHONY: install run debug test test-slow cov lint lint-strict bench charts clean

MYPY_FLAGS = --warn-return-any --warn-unused-ignores \
	--ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

# Install runtime + dev dependencies
install:
	uv sync --group dev

# Run the pipeline with default arguments
run:
	uv run python -m src

# Run the pipeline under the debugger
debug:
	uv run python -m pdb -m src

# Fast unit tests (no model download needed)
test:
	uv run pytest

# Also run the tests that load the real Qwen3-0.6B model
test-slow:
	uv run pytest -m "slow or not slow"

# Unit tests with a coverage report
cov:
	uv run pytest --cov --cov-report=term-missing

# Mandatory lint checks
lint:
	uv run flake8 .
	uv run mypy src benchmarks $(MYPY_FLAGS)

lint-strict:
	uv run flake8 .
	uv run mypy src benchmarks --strict

# Full benchmark (~12 min on CPU): writes data/benchmarks/results.json
bench:
	uv run python -m benchmarks.run_benchmark

# Regenerate the README charts from data/benchmarks/results.json
charts:
	uv run python -m benchmarks.make_charts

# Remove caches
clean:
	rm -rf .mypy_cache .pytest_cache .coverage htmlcov
	find . -type d -name "__pycache__" -not -path "./.venv/*" -exec rm -rf {} +
