# 01 Load inverted list in Rust

## Background

We want to load the inverted list using Rust code.
Right now we don't care about cache but we want to
at least allow parallel search on same index.

The exposed interface on rust side looks like this:

```rust
// Load the index without the cluster data
let path = Path::from("example.ivf");
let index_without_cluster_data = s3.get_range(path, 0..cluster_data_offset).await?;
let index = ffi::create_index(index_without_cluster_data);

// Search from faiss is sync, but the fetching cluster data is async.
// We need to able to block on the fetching cluster data.
let cluster_sizes = index.cluster_sizes();
let fetcher = ClustersFetcher{
    s3: s3, // Just move into ClustersFetch for now, it should be shared
    path: path,
    cluster_data_offset: cluster_data_offset,
    cluster_sizes: cluster_sizes,
};
// TODO: replace the invlist, use the fetcher and free the result
index.search(query, nprobe, k)
```

## Implementation

- [x] return cluster sizes to rust
  - [ ] More efficient for copy from `std::vector` to `rust::Vec`