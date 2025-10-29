# CI: Application image using pre-built base
# Uses: ghcr.io/at15/faiss-s3/ci-base:latest
# Builds: faiss-s3 server only
# Frequency: Every commit
# Build time: ~2-3 minutes

ARG BASE_IMAGE=ghcr.io/at15/faiss-s3/ci-base:latest

FROM ${BASE_IMAGE} AS builder

WORKDIR /build

# Dependencies already exist at /build/deps/ from base image

# Copy application source
COPY CMakeLists.txt config.sh ./
COPY *.cpp *.h ./

# Build server
COPY ci/build-server.sh /tmp/
RUN /tmp/build-server.sh

# Runtime stage
FROM ubuntu:24.04

COPY ci/install-runtime-deps.sh /tmp/
RUN /tmp/install-runtime-deps.sh

WORKDIR /app

COPY --from=builder /build/build/s3_cache_server /app/s3_cache_server
COPY --from=builder /build/deps/aws-sdk-cpp-home/lib /usr/local/lib
COPY --from=builder /build/deps/faiss-home/lib /usr/local/lib

RUN ldconfig

EXPOSE 9001
CMD ["/app/s3_cache_server"]
