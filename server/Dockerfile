# Build stage
FROM ubuntu:24.04 AS builder

# Set environment variable to skip package installation in build scripts
ENV SKIP_APT_INSTALL=1

# Install all build dependencies upfront
RUN apt-get update && apt-get install -y \
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

WORKDIR /build

# Copy dependency build scripts
COPY deps/build-s3-sdk.sh deps/build-s3-sdk.sh
COPY deps/build-faiss.sh deps/build-faiss.sh

# Build AWS SDK C++
RUN cd deps && \
    ./build-s3-sdk.sh && \
    cd ..

# Build FAISS
RUN cd deps && \
    ./build-faiss.sh && \
    cd ..

# Copy source files
COPY CMakeLists.txt .
COPY config.sh .
COPY *.cpp *.h ./

# Configure and build the server
RUN ./config.sh && \
    make -C build -j$(nproc)

# Runtime stage
FROM ubuntu:24.04

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
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

WORKDIR /app

# Copy built server binary
COPY --from=builder /build/build/s3_cache_server /app/s3_cache_server

# Copy dependency libraries
COPY --from=builder /build/deps/aws-sdk-cpp-home/lib /usr/local/lib
COPY --from=builder /build/deps/faiss-home/lib /usr/local/lib

# Update library cache
RUN ldconfig

# Expose default port (adjust if needed)
EXPOSE 9001

# Run the server
CMD ["/app/s3_cache_server"]
