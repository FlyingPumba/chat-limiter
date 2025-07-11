#!/bin/bash

# This ensures the script exits immediately if any command fails, including the tests.
set -e

echo "Starting deployment process..."

# Parse command line arguments
MAJOR_VERSION=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --major)
            MAJOR_VERSION=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: ./deploy.sh [--major]"
            echo "  --major: Bump major version instead of minor"
            exit 1
            ;;
    esac
done

echo "Running tests..."
uv run pytest

# Get current version from pyproject.toml
CURRENT_VERSION=$(grep '^version = ' pyproject.toml | sed 's/version = "\(.*\)"/\1/')
echo "Current version: $CURRENT_VERSION"

# Parse version components
IFS='.' read -ra VERSION_PARTS <<< "$CURRENT_VERSION"
MAJOR=${VERSION_PARTS[0]}
MINOR=${VERSION_PARTS[1]}
PATCH=${VERSION_PARTS[2]}

# Bump version
if [ "$MAJOR_VERSION" = true ]; then
    NEW_MAJOR=$((MAJOR + 1))
    NEW_VERSION="$NEW_MAJOR.0.0"
    echo "Bumping major version: $CURRENT_VERSION -> $NEW_VERSION"
else
    NEW_MINOR=$((MINOR + 1))
    NEW_VERSION="$MAJOR.$NEW_MINOR.0"
    echo "Bumping minor version: $CURRENT_VERSION -> $NEW_VERSION"
fi

# Update version in pyproject.toml
sed -i.bak "s/version = \"$CURRENT_VERSION\"/version = \"$NEW_VERSION\"/" pyproject.toml
rm pyproject.toml.bak

echo "Version bumped to: $NEW_VERSION"

echo "Re-generating uv lockfile..."
uv lock

# Commit and push changes
echo "Committing version bump..."
git add pyproject.toml
git add uv.lock
git commit -m "Bump version to $NEW_VERSION"

echo "Pushing to remote..."
git push

echo "Building project..."
[ -d dist ] && rm -r dist
uv run python -m build

echo "Publishing to PyPI..."
uv run python -m twine upload --repository pypi --non-interactive dist/* -u "__token__" -p "$PYPI_API_KEY"

echo "Deployment completed successfully!"