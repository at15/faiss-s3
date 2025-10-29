#!/bin/sh

AWS_SDK_INSTALL_DIR="$(realpath deps/aws-sdk-cpp-home)"

# TODO: this only works on mac with openmp
cmake -B build \
  -DOpenMP_C_FLAGS="-Xclang -fopenmp -I/opt/homebrew/opt/libomp/include" \
  -DOpenMP_C_LIB_NAMES="omp" \
  -DOpenMP_omp_LIBRARY=/opt/homebrew/opt/libomp/lib/libomp.dylib \
  -DOpenMP_CXX_FLAGS="-Xclang -fopenmp -I/opt/homebrew/opt/libomp/include" \
  -DOpenMP_CXX_LIB_NAMES="omp" \
  -DCMAKE_PREFIX_PATH="$AWS_SDK_INSTALL_DIR" \
  .