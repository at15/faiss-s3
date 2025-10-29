#!/bin/sh

set -e

cd "$(dirname "$0")"

mkdir -p faiss-home
FAISS_INSTALL_DIR="$(realpath faiss-home)"

git clone --depth 1 https://github.com/facebookresearch/faiss.git
cd faiss

# Detect operating system
OS="$(uname -s)"

case "$OS" in
  Darwin*)
    echo "Detected macOS, using Homebrew..."
    brew install libomp

    # Configure with macOS-specific OpenMP settings
    cmake -B build \
      -DCMAKE_BUILD_TYPE=Release \
      -DFAISS_ENABLE_GPU=OFF \
      -DFAISS_ENABLE_PYTHON=OFF \
      -DFAISS_ENABLE_EXTRAS=OFF \
      -DBUILD_TESTING=OFF \
      -DOpenMP_omp_LIBRARY=/opt/homebrew/opt/libomp/lib/libomp.dylib \
      -DOpenMP_CXX_FLAGS="-Xclang -fopenmp -I/opt/homebrew/opt/libomp/include" \
      -DOpenMP_CXX_LIB_NAMES="omp" \
      -DCMAKE_INSTALL_PREFIX="$FAISS_INSTALL_DIR" \
      .
    ;;

  Linux*)
    echo "Detected Linux, using apt..."
    sudo apt-get update
    # https://github.com/facebookresearch/faiss/wiki/Troubleshooting#surprising-faiss-openmp-and-openblas-interaction
    sudo apt-get install -y libopenblas-openmp-dev

    # Configure with standard Linux settings
    cmake -B build \
      -DCMAKE_BUILD_TYPE=Release \
      -DFAISS_ENABLE_GPU=OFF \
      -DFAISS_ENABLE_PYTHON=OFF \
      -DFAISS_ENABLE_EXTRAS=OFF \
      -DBUILD_TESTING=OFF \
      -DCMAKE_INSTALL_PREFIX="$FAISS_INSTALL_DIR" \
      .
    ;;

  *)
    echo "Unsupported operating system: $OS"
    exit 1
    ;;
esac

make -C build install

echo "FAISS installed successfully to $FAISS_INSTALL_DIR"