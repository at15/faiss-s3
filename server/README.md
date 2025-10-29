# Server

## Usage

```bash
# TODO: Add how to build faiss from source (at least on mac...), also need openmp

./deps/build-s3-sdk.sh
make config
make build

# Start s3mock server
export AWS_ACCESS_KEY_ID=test
export AWS_SECRET_ACCESS_KEY=test
export AWS_REGION=us-east-1
export AWS_EC2_METADATA_DISABLED=true
export S3_ENDPOINT_URL=http://localhost:9000
export AWS_ENDPOINT_URL_S3=http://localhost:9000
# Run in foreground, listening on port 9001
./build/s3_cache_server
```

Ubuntu

- [ ] Build FAISS from source, conda is too old
- `#include <signal.h>` on linux hmm

```bash
# For FAISS
sudo apt install -y libopenblas-openmp-dev

# For AWS SDK
sudo apt install -y \
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

```