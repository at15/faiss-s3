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

C++ and Rust

```bash
# generate the rust binding, it would fail first time due to missing the cpp lib
cargo build

# Create the cmake build directory
# Depends on cargo build generated rust/cxx.h, lib.rs.h, lib.rs.cc
make config

# Build the cpp library
make build

# Build the rust binary
cargo build
./target/debug/faiss-s3-rs
```

Python

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install maturin
```

Run the python REPL

```python
import faiss_s3_rs
print(faiss_s3_rs.sum_as_string(1, 2))
faiss_s3_rs.create_example_ivf_index("frompy.ivf")
```