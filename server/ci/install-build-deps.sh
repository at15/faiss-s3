#!/bin/bash
# Install build-time dependencies
# Used by: Dockerfile, ci/Dockerfile.base

set -e

echo "Installing build dependencies..."

apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libssl-dev \
    libcurl4-openssl-dev \
    libz-dev \
    libbz2-dev \
    liblz4-dev \
    libzstd-dev \
    libxml2-dev \
    libuuid1 \
    uuid-dev \
    libopenblas-openmp-dev \
    && rm -rf /var/lib/apt/lists/*

echo "✓ Build dependencies installed successfully"
