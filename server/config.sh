#!/bin/sh

set -e

# Change to the directory where this script is located
cd "$(dirname "$0")"

AWS_SDK_INSTALL_DIR="$(realpath deps/aws-sdk-cpp-home)"
FAISS_INSTALL_DIR="$(realpath deps/faiss-home)"

# Detect operating system
OS="$(uname -s)"

case "$OS" in
  Darwin*)
    echo "Configuring for macOS..."
    cmake -B build \
      -DOpenMP_C_FLAGS="-Xclang -fopenmp -I/opt/homebrew/opt/libomp/include" \
      -DOpenMP_C_LIB_NAMES="omp" \
      -DOpenMP_omp_LIBRARY=/opt/homebrew/opt/libomp/lib/libomp.dylib \
      -DOpenMP_CXX_FLAGS="-Xclang -fopenmp -I/opt/homebrew/opt/libomp/include" \
      -DOpenMP_CXX_LIB_NAMES="omp" \
      -DCMAKE_PREFIX_PATH="$AWS_SDK_INSTALL_DIR;$FAISS_INSTALL_DIR" \
      .
    ;;

  Linux*)
    echo "Configuring for Linux..."
    cmake -B build \
      -DCMAKE_PREFIX_PATH="$AWS_SDK_INSTALL_DIR;$FAISS_INSTALL_DIR" \
      .
    ;;

  *)
    echo "Unsupported operating system: $OS"
    exit 1
    ;;
esac

echo "Configuration complete. Run 'make -C build' to build."
