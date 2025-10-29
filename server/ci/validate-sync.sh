#!/bin/bash
# Validate that Dockerfiles use consistent dependencies
# Prevents drift between local and CI builds

set -e

# Change to the directory where this script is located, then to parent (server/)
cd "$(dirname "$0")/.."

echo "Validating Docker dependency consistency..."
echo "Working directory: $(pwd)"

# Check that install-build-deps.sh is used by Dockerfile
if ! grep -q "install-build-deps.sh" Dockerfile; then
    echo "ERROR: Dockerfile doesn't use install-build-deps.sh"
    exit 1
fi

# Check that install-build-deps.sh is used by ci/Dockerfile.base
if ! grep -q "install-build-deps.sh" ci/Dockerfile.base; then
    echo "ERROR: ci/Dockerfile.base doesn't use install-build-deps.sh"
    exit 1
fi

# Check that install-runtime-deps.sh is used by Dockerfile
if ! grep -q "install-runtime-deps.sh" Dockerfile; then
    echo "ERROR: Dockerfile doesn't use install-runtime-deps.sh"
    exit 1
fi

# Check that install-runtime-deps.sh is used by ci/Dockerfile.app
if ! grep -q "install-runtime-deps.sh" ci/Dockerfile.app; then
    echo "ERROR: ci/Dockerfile.app doesn't use install-runtime-deps.sh"
    exit 1
fi

echo "✓ Docker dependency sync validated"
