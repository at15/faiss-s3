#!/bin/bash
# Install runtime-only dependencies
# Used by: Dockerfile (runtime stage), ci/Dockerfile.app (runtime stage)

set -e

echo "Installing runtime dependencies..."

apt-get update && apt-get install -y \
    libssl3t64 \
    libcurl4 \
    zlib1g \
    libbz2-1.0 \
    liblz4-1 \
    libzstd1 \
    libxml2 \
    libuuid1 \
    libopenblas0-openmp \
    && rm -rf /var/lib/apt/lists/*

echo "✓ Runtime dependencies installed successfully"
