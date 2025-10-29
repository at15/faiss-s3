# Build stage
FROM ubuntu:24.04 AS builder

# Set environment variable to skip package installation in build scripts
ENV SKIP_APT_INSTALL=1

# Install build dependencies using shared script
COPY ci/install-build-deps.sh /tmp/
RUN /tmp/install-build-deps.sh

WORKDIR /build

# Build dependencies
COPY deps/build-s3-sdk.sh deps/build-s3-sdk.sh
COPY deps/build-faiss.sh deps/build-faiss.sh
RUN cd deps && ./build-s3-sdk.sh
RUN cd deps && ./build-faiss.sh

# Copy application source
COPY CMakeLists.txt config.sh ./
COPY *.cpp *.h ./

# Build server using shared script
COPY ci/build-server.sh /tmp/
RUN /tmp/build-server.sh

# Runtime stage
FROM ubuntu:24.04

# Install runtime dependencies using shared script
COPY ci/install-runtime-deps.sh /tmp/
RUN /tmp/install-runtime-deps.sh

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
