# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

faiss-s3 enables on-demand Faiss vector search from S3 by lazy-loading clusters without fetching the entire index. The system consists of:

- **C++ Server**: TCP server that loads Faiss IVF indexes from S3 and performs searches with on-demand cluster caching
- **Python Client**: TCP client library for communicating with the server
- **Custom Faiss Extension**: S3OnDemandInvertedLists implements on-demand cluster loading with LRU caching

## Architecture

### Server Components (`server/`)

The server implements a custom Faiss InvertedLists that fetches cluster data from S3 on demand:

- **S3InvertedLists.h/cpp**: Core implementation
  - `S3ReadNothingInvertedLists`: Placeholder during index loading
  - `S3OnDemandInvertedLists`: Main implementation with S3 client integration and LRU cache
  - Uses AWS SDK C++ S3CrtClient for range requests
  - Integrates with Faiss via custom IO flag `IO_FLAG_S3`

- **s3_cache_server.cpp**: TCP server handling:
  - `LOAD`: Load index metadata from S3, setup on-demand loading
  - `SEARCH`: Perform k-NN search, auto-fetch clusters as needed
  - `INFO`: Return cache statistics
  - `ECHO`: Connection testing
  - Binary protocol: text commands + length-prefixed binary data

### Client Components (`client/`)

- **src/faiss_s3/client.py**: Python client implementing the TCP protocol
  - Context manager for connection handling
  - Methods: `load()`, `search()`, `info_cache()`, `info_index()`, `echo()`
  - Uses numpy for vector handling

### Dependencies

The project depends on vendored dependencies in `server/deps/`:
- **aws-sdk-cpp**: Built with S3-CRT client for high-performance range requests
- **faiss**: Facebook's vector search library

## Build Commands

### Server Build

The server uses CMake and requires pre-built dependencies:

```bash
# Build dependencies (run once)
cd server/deps
./build-s3-sdk.sh    # Build AWS SDK C++
./build-faiss.sh     # Build Faiss

# Configure server (macOS/Linux detected automatically)
cd server
./config.sh

# Build server
make build

# Format code
make format
```

### Docker Build

Docker provides a reproducible build environment:

```bash
# Build Docker image (Ubuntu 24.04 based)
cd server
make docker-build

# Run server with S3Mock access
make docker-run

# Interactive shell in container
make docker-shell
```

The Docker build is multi-stage:
1. Builder stage: Installs dependencies, builds AWS SDK, Faiss, and server
2. Runtime stage: Copies binaries and required shared libraries

### Python Client

```bash
# Install in development mode
python3 -m venv .venv
source .venv/bin/activate
pip install -e ./client

# Install optional dependencies for tests
pip install sentence-transformers pandas
```

## Testing

```bash
# Test imports and basic functionality
python tests/test_import.py

# Integration test with S3 cache server
python tests/test_s3_cache_server.py

# End-to-end test with Quora dataset
python tests/test_quora.py
```

Tests are located in `tests/` directory and use the faiss_s3 client package.

## Configuration

### Server Environment Variables

- `S3_ENDPOINT_URL` / `AWS_ENDPOINT_URL_S3`: Custom S3 endpoint (for S3Mock, MinIO)
- `AWS_REGION`: AWS region (default: us-east-1)
- `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`: AWS credentials
- `AWS_EC2_METADATA_DISABLED`: Set to "true" when not running on EC2

### Protocol Details

The server uses port 9001 by default. Protocol:
- Commands: Space-separated `key=value` pairs, newline-terminated
- Binary data: 4-byte little-endian length prefix + raw bytes
- Responses follow same format, errors start with "ERROR"

## Key Implementation Details

### S3 Range Requests

Cluster data is fetched using HTTP range requests:
```
Range: bytes=start-end
```

The cluster offset is calculated from:
1. Base cluster_data_offset (provided during LOAD)
2. Cumulative sizes of preceding clusters
3. Each cluster contains codes (uint8_t) and IDs (int64_t)

### Faiss IO Hook

The project registers a custom Faiss IO hook using `IO_FLAG_S3`:
- During `read_index()`, the hook intercepts InvertedLists creation
- Creates `S3ReadNothingInvertedLists` placeholder with cluster sizes
- After loading, replaced with `S3OnDemandInvertedLists` for actual operation

### Cache Management

LRU cache with configurable size limit:
- Tracks hits/misses per index and globally
- Evicts least-recently-used clusters when memory limit reached
- Statistics available via INFO command

## Platform-Specific Notes

### macOS

OpenMP configuration required (Homebrew libomp):
```bash
# config.sh handles this automatically
-DOpenMP_C_FLAGS="-Xclang -fopenmp -I/opt/homebrew/opt/libomp/include"
-DOpenMP_omp_LIBRARY=/opt/homebrew/opt/libomp/lib/libomp.dylib
```

### Linux

Standard OpenMP from system packages works without special configuration.

### Ubuntu 24.04

The Docker build targets Ubuntu 24.04 which includes:
- libopenblas-openmp-dev for BLAS
- Modern C++ toolchain (gcc-13/g++-13)
- CMake 3.28+

## Common Workflows

### Adding a new server command

1. Update protocol handler in `s3_cache_server.cpp`
2. Add response parsing in `client/src/faiss_s3/client.py`
3. Update protocol documentation in client docstrings
4. Add test case in `tests/test_s3_cache_server.py`

### Modifying S3 inverted lists behavior

1. Update `S3OnDemandInvertedLists` in `server/S3InvertedLists.cpp`
2. Rebuild server: `cd server && make build`
3. Test with integration tests

### Debugging S3 requests

Enable server logging by checking stdout/stderr. The server logs:
- S3 endpoint configuration
- Index load operations
- Search requests and cache statistics
- S3 GetObject operations (on errors)