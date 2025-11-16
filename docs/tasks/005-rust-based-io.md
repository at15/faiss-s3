---
tags:
  - rust
  - cpp
  - io
  - http
---

# 005 Rust Based IO

Use rust to make network requests and provide http server.
Futher more, we can [create python binding using rust pyo3](./006-python-binding-via-rust.md).

## Background

Currently we are using AWS C++ SDK to make S3 requests. It has a few issues:

- It only works with S3, it does not work with presigned url, does not work GCS.
  - For rust there is crate like https://github.com/apache/arrow-rs-object-store that is suppose to work for all object storages
    - It is not importing invidual cloud's SDK and implemented the auth logic
- Creating a HTTP(s) server is a bit painful, there are many libraries and we endup having our own TCP based protocol
- I don't know how to create python binding, I saw faiss using swig, but I feel pyo3 is better
- Package management is a lot better in rust compared to C++
  - We plan to add [full text search using tantivy](./004-full-text-search.md)

The rough plan right now is

- Create a exp folder so we can keep existing C++ code and save the experiment code
- Use python binding to call rust code to call C++ code, i.e we don't need a http server in the beginning
- Skip caching logic and simply make a new S3 call for each cluster
- Make sure the code compiles and work e2e, then we can start on
  - Implement the caching logic in rust
  - Implement the http server in rust
  - Integrate full text search and [filter](./003-filter.md) on metadata

## Implementation

- [ ] search local faiss file using python binding calling C++
  - this is does not even require our customized inverted list build
  - [x] create example index works
- [x] When using CMake, how does C++ side link the rust implementation during build? ... Or it is linking a rust static library? If we are building a library in C++, it only requires the header and does not do the linking unless we are building a executable.
- [ ] hook the network request logic for inverted list using rust, no cache (for now), just query and forget