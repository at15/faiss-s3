#!/bin/sh

set -e

# Change to the directory where this script is located
cd "$(dirname "$0")"

# FAISS_INSTALL_DIR="$(realpath deps/faiss-home)"

echo "Configuring for macOS..."
cmake -B build \
    -DOpenMP_C_FLAGS="-Xclang -fopenmp -I/opt/homebrew/opt/libomp/include" \
    -DOpenMP_C_LIB_NAMES="omp" \
    -DOpenMP_omp_LIBRARY=/opt/homebrew/opt/libomp/lib/libomp.dylib \
    -DOpenMP_CXX_FLAGS="-Xclang -fopenmp -I/opt/homebrew/opt/libomp/include" \
    -DOpenMP_CXX_LIB_NAMES="omp" \
    .