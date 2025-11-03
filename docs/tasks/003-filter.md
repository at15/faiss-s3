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

We can start with YFCC 100M because the data is small and there is ground truth.

## YFCC 100M

From NIPS2023 BigANN Benchmark https://github.com/harsha-simhadri/big-ann-benchmarks/blob/main/neurips23/README.md

- https://github.com/harsha-simhadri/big-ann-benchmarks/blob/main/benchmark/datasets.py

btw: TBH I think the tags are very limited and not realistic.
For example, when I am searching for code, I could have combined filter like `lang:python AND path:comp_a/comp_c/*`

This makes the baseline using faiss has good recall because either metadata
first or IVF first would work ...

```bash
git clone https://github.com/harsha-simhadri/big-ann-benchmarks.git
cd big-ann-benchmarks
# The instruction was saying using 3.10 ... I will try 3.13 and see if it works
# python3.13 -m venv .venv
# 3.13 does NOT work ...
brew install python@3.12
python3.12 -m venv .venv
pip install -r requirements_py3.12.txt
python create_dataset.py --help

# Download from a s3 bucket downloading 
# https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/yfcc100M/base.10M.u8bin -> data/yfcc100M/base.10M.u8bin...
#   [44.32 s] downloaded 1586.00 MiB / 1831.05 MiB at 35.78 MiB/s
# Total 2.7GB data
python create_dataset.py --dataset yfcc-10M
```

The run instruction need to use its docker image and/or `run.py`.
Which is not convinement for our testing here...

The faiss baseline for YFCC 100M is doing the following:

- Build index on tags as a sparse matrix, the logic is in the [bow_id_selector.swig](https://github.com/harsha-simhadri/big-ann-benchmarks/blob/590a7261af1cc4c0d615e43a3d30ba916d852476/neurips23/filter/faiss/bow_id_selector.swig#L60-L104)
 - total 200k tags from vocabulary
 - each row index is the index of the vector
 - each column means if the vector has that specific tag
- Determine order base on frequency of tag
  - If low, filter by tag first and kNN. 
  - If high, ANN using IVF and pass the bow id selector to `faiss.SearchParameters(sel=make_bow_id_selector(selector))`

When frequency is low

- First fetch all the vectors that match the tags, two tag is intersection on the sorted vector indices
- Then do kNN on the subset

```python
# Lines 224-235
if freq < self.metadata_threshold:  # default: 0.001
    # Find all docs with required tags
    docs = csr_get_row_indices(docs_per_word, w1)
    if w2 != -1:
        docs = intersect_sorted(docs, csr_get_row_indices(docs_per_word, w2))
    # Compute exact k-NN on this subset
    xb_subset = self.xb[docs]
    _, Ii = faiss.knn(X[q:q+1], xb_subset, k=k)
    self.I[q, :] = docs[Ii.ravel()]
```

When frequency is high

- Pass the bow id selector to `faiss.SearchParameters(sel=make_bow_id_selector(selector))`
- There is optimization Binary Signature ... I will skip for now

When faiss applies the id selector, it will run it after each vector before calcuating the distance
[scan_codes in IndexIVFFlat.cpp](https://github.com/facebookresearch/faiss/blob/f983e3ab69c46ecb7728395613ace22729bbd4a8/faiss/IndexIVFFlat.cpp#L189-L191)

Other IVF types does apply the id selector, though they may not use `scan_codes`

```cpp
size_t scan_codes(
        size_t list_size,
        const uint8_t* codes,
        const idx_t* ids,
        float* simi,
        idx_t* idxi,
        size_t k) const override {
    const float* list_vecs = (const float*)codes;
    size_t nup = 0;
    for (size_t j = 0; j < list_size; j++) {
        const float* yj = list_vecs + vd.d * j;
        if (use_sel && !sel->is_member(ids[j])) {
            continue;
        }
        float dis = vd(xi, yj);
        if (C::cmp(simi[0], dis)) {
            int64_t id = store_pairs ? lo_build(list_no, j) : ids[j];
            heap_replace_top<C>(k, simi, idxi, dis, id);
            nup++;
        }
    }
    return nup;
}
```

In `IVFPQ` it is [skip_entry](https://github.com/facebookresearch/faiss/blob/11904f69817fb7cd93028b327abf7f267daf7ad1/faiss/IndexIVFPQ.cpp#L993-L995)

```cpp
for (size_t j = 0; j < ncode; j++, codes += pq.code_size) {
    if (res.skip_entry(j)) {
        continue;
    }
    pq.decode(codes, decoded_vec);
}
```

If we want to control the cluster, we should use `search_preassigned` or pass ID selector as parameter to the quantizer