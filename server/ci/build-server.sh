#!/bin/bash
# Configure and build the server binary
# Used by: Dockerfile, ci/Dockerfile.app

set -e

echo "Configuring server build..."
./config.sh

echo "Building server..."
make -C build -j$(nproc)

echo "✓ Server built successfully"
