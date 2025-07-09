#!/bin/bash

# This ensures the script exits immediately if any command fails, including the tests.
set -e

echo "Starting deployment process..."

echo "Running tests..."
uv run pytest

echo "Building project..."
uv run python -m build

echo "Publishing to PyPI..."
uv run python -m twine upload --repository pypi --non-interactive dist/* -u "__token__" -p "$PYPI_API_KEY"

echo "Deployment completed successfully!"