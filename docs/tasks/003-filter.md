---
tags:
  - filter
  - metadata
  - search
  - benchmark
  - ann
---

# Filter on metadata

Allow user to filter on metadata.

## Links

- https://turbopuffer.com/blog/native-filtering
- https://github.com/lancedb/tantivy-object-store I guess Lance switched to writing their own index format in the end

## Background

When we are only using faiss and parquet file, we can only filter in two ways:

- Post filter, first query faiss, then fetch attributes from parquet and filter out result that does not match
- Pre filter, first filter attributes from parquet, then do kNN on all the results and got no benefits from the ANN index

What the turbopuffer blog talks about 'Native filtering` is one thing:

Save vector index info `{cluster_id}:{local_id}` in the attributes index to prune the clusters in ANN.
There is also a down sampled version that only keeps `{cluster_id}`, it is kind of like doc id without position.

In turbopuffer, they consider update, for faiss-s3, we only support replacing entire set of documents so there is no in place update we are considering.

In our design, we can use a existing full text search engine such as tanvity/lucene and save the cluster id in the full text search index as its document id.

Things we need to consider:

- Can FAISS specify the cluster ids when searching the index?
- Implement object storage directory to fetch from object storage to memory directly https://github.com/lancedb/tantivy-object-store
- See if we can update our faiss C++ code to use Rust, this would allow us to remove the C++ cache and TCP server logic

What we can do now:

- Find a dataset
- See if we can get it working using faiss and tantivy using local file via python binding
- Use rust code to glue C++, Tanvity together and provide python binding using PyO3

## Datasets

- https://github.com/qdrant/ann-filtering-benchmark-datasets random
  - `all-MiniLM-L6-v2` 2M arxiv
  - LAION CLIP, 100k
    - Full LAION is 400M?
- LAION 5B
  - https://github.com/rom1504/img2dataset/blob/main/dataset_examples/laion5B.md
  - https://rom1504.medium.com/semantic-search-at-billions-scale-95f21695689a uses Faiss, the code is https://github.com/rom1504/clip-retrieval
- https://github.com/pinecone-io/VSB they have a list of public datasets on GCS https://docs.pinecone.io/guides/data/use-public-pinecone-datasets
  - Which does not have the yfc10M dataset they mentioned ...

We can start with YFCC 100M because the data is small and there is ground truth.
See [YFCC100M.md](../dataset/YFCC100M.md) for more details.

## Design

- Use filter + kNN to calculate ground truth
- Use a smaller subset of data to get started

## Implementation

- [ ] Use local file system with faiss-cpu and tantivy-py
  - btw: I think it can be slower (maybe much slower) than the faiss baseline ...