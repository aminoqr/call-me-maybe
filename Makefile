# Makefile for Call Me Maybe project

# The installation command using uv
install:
		uv sync

# Run the project with default arguments
run:
		uv run python -m src

# Run the project in debug mode
debug:
		uv run python -m pdb -m src
	
# Clean up temporary Python files and caches
clean:
		rm -rf __pycache__ .mypy_cache .pytest_cache
		find . -type d -name "__pycache__" -exec rm -rf {} +

# Run the mandatory linting checks
lint:
		uv run flake8 .
		uv run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
		uv run flake8 .
		uv run mypy . --strict