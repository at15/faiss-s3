---
tags:
  - dataset
  - filter
  - yfcc100m
---

# YFCC 100M

Used by big ANN benchmark, baseline is using faiss. The big ANN benchmark only fiters by two tags.

## Links

From NIPS2023 BigANN Benchmark https://github.com/harsha-simhadri/big-ann-benchmarks/blob/main/neurips23/README.md

- https://github.com/harsha-simhadri/big-ann-benchmarks/blob/main/benchmark/datasets.py
- Detail of dataset is in the claude generated document [YFCC100M.claude.md](YFCC100M.claude.md)

## Setup

The **Ground Truth** is calculated using filter by metadata then kNN, run on GPU using slurm... saw `srun`.

btw: I think the tags are very different from code search, e.g. there can be combined and wildcard filter like `lang:python AND path:comp_a/comp_c/*`

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

## Faiss Baseline

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

Actually I kind of want to see the original image, but seems the dataset does not have the originl image, also the vector is quantized to 8bit.
We can probably try https://huggingface.co/datasets/dalle-mini/YFCC100M_OpenAI_subset

## Read dataset

Well ... I should just read the code and README, they are a lot cleaner than claude code's summary

```python
class YFCC100MDataset(DatasetCompetitionFormat):
    """ the 2023 competition """

    def __init__(self, filtered=True, dummy=False):
        self.filtered = filtered
        nb_M = 10 # number of base vectors in millions
        self.nb_M = nb_M
        self.nb = 10**6 * nb_M # 10 million base vectors
        self.d = 192
        self.nq = 100000
        self.dtype = "uint8"
```