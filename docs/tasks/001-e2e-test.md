---
tags:
  - test
  - e2e
  - faiss
  - search
---

# 001 E2E Test

Full e2e test from raw text without hardcoded values.

## Background

Right now we have multiple tests and each of them are kind of incomplete.
For example

- `test_quora.py` generates embedding, index, upload to s3 but it only queries the index on local file system and is not using our s3 cache server
- `test_s3_cache_server.py` queries the index from s3 cache server but it using hardcoded metadata such as `cluster_data_offset`

Further more, we need to comeup with the storage layout on S3 because a
full search experience also allow retrieving the source text and additional attributes.

Let's first look at the full e2e flow (from a user perspective)

- Download the dataset, e.g. http, huggingface
- Create embeddings, e.g. use SentenceTransformer, call OpenAI API
- Create index
- Generate metata from index, `faiss_s3/meta.py` has the logic to calculate the cluster data offset so the cache server can fetch the cluster data on demand, `test_meta.py` has example usage with hard coded value
- Upload files (index, attributes, meta) to S3, embedding is optional, it is useful if we want to create new index.
- Query the index, there are two ways
  - Download the full file and query locally by loading it via mmap
  - Use the faiss_s3 client to query the index from S3 cache server, have the cache server to load the index on demand and cache recently used clusters.

The layout on S3 looks something like this:

```text
s3://test-bucket/quora/
    - meta.json
    - index.ivf
    - attributes.parquet
    - embeddings.npy
```

The `meta.json` should save information on both the index and other files.
This allow inspecting the scale of index without having to download index/parquet/embedding file.
For example

```json
{
    "version": "1",
    "index": {
        "file": "index.ivf",
        "size_bytes": 1000000,
        "type": "IndexIVFFlat",
        "vector_size": 768,
        "n_vectors": 1000000,
        "n_clusters": 1024,
        "cluster_data_offset": 800000,
    },
    "attributes": {
        "file": "attributes.parquet",
        "size_bytes": 1000000,
        "columns": [
            {
                "name": "source",
                "type": "string"
            },
            {
                "name": "file",
                "type": "string"
            }
        ]
    },
    "embedding": {
        "file": "embeddings.npy",
        "library": "sentence-transformers",
        "model": "quora-distilbert-multilingual",
        "vector_size": 768,
        "n_vectors": 1000000,
    }
}
```

## Instructiosn for Claude Code on version 1

Base on the background section and existing scripts in `tests` folder.
Write the plan for the new full e2e test in `docs/tasks/001-e2e-test.claude.md`.
Then implement the new e2e test usign the quora dataset.