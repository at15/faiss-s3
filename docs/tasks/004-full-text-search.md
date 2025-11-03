# Full Text Search

Follw up on [filter](003-filter.md) we can support full text search when we are using full text search engine to implement filtering on metadata.

## Links

Tanvity

- https://github.com/quickwit-oss/tantivy/issues/815
  - They use faiss `IVFFlat` and map cluster, doc id etc. https://github.com/quickwit-oss/tantivy/issues/815#issuecomment-1511964728
  - https://github.com/nuclia/nucliadb not sure about the storage structure though
- https://github.com/quickwit-oss/tantivy/pull/1067 WASM with range request