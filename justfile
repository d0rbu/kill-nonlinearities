# kill-nonlinearities task runner.
# Requires `uv` (https://docs.astral.sh/uv/) and `just` (https://just.systems).
# Run `just` with no arguments to list available recipes.

# Show available recipes.
default:
    @just --list

# Create the dev environment and install git hooks.
setup: install hooks

# Sync the virtual environment from the lockfile (torch + dev tools).
install:
    uv sync

# Install the git pre-commit hooks.
hooks:
    uv run pre-commit install

# Auto-format code and apply safe lint fixes.
fmt:
    uv run ruff format
    uv run ruff check --fix

# Lint and check formatting without writing changes (CI-style).
lint:
    uv run ruff check
    uv run ruff format --check

# Static type check with ty.
typecheck:
    uv run ty check

# Run the test suite. Extra args pass through, e.g. `just test -k version`.
test *args:
    uv run pytest {{args}}

# Run tests and write an HTML coverage report to htmlcov/.
cov:
    uv run pytest --cov-report=html --cov-report=term-missing

# Run every pre-commit hook against all files.
hooks-all:
    uv run pre-commit run --all-files

# The full local gate: lint, type-check, then test.
check: lint typecheck test

# Update the lockfile to the latest allowed versions.
update:
    uv lock --upgrade

# Remove caches and build/coverage artifacts.
clean:
    rm -rf .pytest_cache .ruff_cache .ty_cache .hypothesis htmlcov .coverage coverage.xml dist build
