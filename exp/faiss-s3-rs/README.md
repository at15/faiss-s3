---
tags:
  - exp
  - rust
  - cpp
  - python
  - faiss
  - s3
  - io
  - http
---

# Faiss S3 Rust

Use Rust to provide network requests and python binding.

- `cxx` crate for interop between Rust and C++.
- `pyo3` crate for python binding.

## Build

```sh
# generate the rust binding, it would fail first time due to missing the cpp lib
cargo build

# Create the cmake build directory
make config

# Build the cpp library
make build

# Build the rust binary
cargo build
./target/debug/faiss-s3-rs
```