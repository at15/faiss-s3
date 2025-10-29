#!/bin/sh

set -e

# Change to the directory where this script is located
cd "$(dirname "$0")"

mkdir -p aws-sdk-cpp-home
AWS_SDK_INSTALL_DIR="$(realpath aws-sdk-cpp-home)"

# Detect operating system
OS="$(uname -s)"

case "$OS" in
  Darwin*)
    echo "Detected macOS, using Homebrew..."
    brew install cmake git openssl curl zlib
    ;;

  Linux*)
    echo "Detected Linux, installing dependencies..."
    sudo apt-get update
    sudo apt-get install -y \
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
      uuid-dev
    ;;

  *)
    echo "Unsupported operating system: $OS"
    exit 1
    ;;
esac

git clone --depth 1 --recurse-submodules --shallow-submodules https://github.com/aws/aws-sdk-cpp
cd aws-sdk-cpp
mkdir -p build
cd build

# Build only the S3 CRT client to save time
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DBUILD_ONLY="s3-crt" \
         -DCMAKE_INSTALL_PREFIX="$AWS_SDK_INSTALL_DIR" \
         -DENABLE_TESTING=OFF

# Use nproc on Linux, sysctl on macOS for parallel jobs
if command -v nproc > /dev/null 2>&1; then
  JOBS=$(nproc)
else
  JOBS=$(sysctl -n hw.ncpu 2>/dev/null || echo 4)
fi

make -j"$JOBS"
make install

echo "AWS SDK C++ installed successfully to $AWS_SDK_INSTALL_DIR"