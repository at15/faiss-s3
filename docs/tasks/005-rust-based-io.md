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

## Read from S3 without cache

Let's first do it from rust, it is a bit easier without python, plus we can run a standalone rust server.

The rough steps are

- Open a index with parameters
  - object path
  - offsets of the clusters
- Search

The pseudo code looks like this:

```text
// From rust side
config = {
  object_path: "s3://bucket/path/to/index.ivf",
  n_clusters: 1024,
  cluster_data_offset: 123456,
}

data_without_invlist = bucket.get_range(object_path, [0, cluster_data_offset])
// C++ code using the S3ReadNothingInvList to get the number of vectors in each cluster
// Pass the callback to trigger the rust function to fetch the inverted list data
faiss_index = ffi::read_index(data_without_invlist, bucket.get_range_callback)
faiss_index.search(query_vectors, top_k)
```

- C++ side should be using `unique_ptr` so it drops the index when rust side drop it

Steps

- [ ] make sure the `object_store` crate works for s3 and presigned url
- [ ] define the interface for passing the callback
- [ ] test the example ivf file
- [ ] compare the result, we can use the same quora example, though the query still need embedding generated from python code.