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