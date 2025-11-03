# YFCC-100M Filter Track Dataset Documentation

## Table of Contents

- [Overview](#overview)
- [Dataset Files](#dataset-files)
- [File Format Specifications](#file-format-specifications)
  - [u8bin Format (Vectors)](#u8bin-format-vectors)
  - [spmat Format (Sparse Metadata)](#spmat-format-sparse-metadata)
  - [ibin Format (Ground Truth)](#ibin-format-ground-truth)
- [Reading the Data](#reading-the-data)
- [Dataset Characteristics](#dataset-characteristics)
- [Understanding the Metadata](#understanding-the-metadata)
- [Implementing Filtered Search](#implementing-filtered-search)
  - [Strategy 1: Post-Filtering](#strategy-1-post-filtering)
  - [Strategy 2: Pre-Filtering](#strategy-2-pre-filtering)
  - [Strategy 3: Hybrid Approach](#strategy-3-hybrid-approach-faiss-baseline)
- [Evaluation](#evaluation)
- [Complete Example](#complete-example)
- [Performance Considerations](#performance-considerations)

## Overview

The YFCC-100M dataset is part of the NeurIPS 2023 Big-ANN Benchmarks Filter Track. It consists of:

- **10 million** image embeddings extracted from YFCC100M dataset using CLIP (Zilliz's CLIP descriptors)
- **100,000** query embeddings (public test set)
- **Metadata tags** associated with each image (camera model, year, country, description words)
- **Filtered search task**: Find k nearest neighbors that match specific metadata tags

**Key Challenge**: Given a query vector and 1-2 required tags, find the k nearest neighbors among vectors that have ALL the required tags.

**Important Dataset Characteristics**:
- **Subset**: 10M vectors from the full YFCC100M dataset (which has 100M images)
- **Embeddings**: CLIP-based, dimension reduced/quantized to 192-dim
- **Data Type**: **uint8** (8-bit unsigned integers, NOT float32!)
- **Distance**: Euclidean (L2)
- **Size**: ~1.9 GB (vectors) + ~400 MB (metadata)

**Download URL**: `https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/yfcc100M/`

## Dataset Files

| File | Size | Format | Description |
|------|------|--------|-------------|
| `base.10M.u8bin` | ~1.9GB | u8bin | 10M CLIP embeddings (192-dim, uint8) |
| `query.public.100K.u8bin` | ~19MB | u8bin | 100K query vectors |
| `base.metadata.10M.spmat` | ~400MB | spmat | Sparse metadata matrix (10M × 200K) |
| `query.metadata.public.100K.spmat` | ~4MB | spmat | Query filter tags (100K × 200K) |
| `GT.public.ibin` | ~40MB | ibin | Ground truth k-NN results (filtered) |
| `unfiltered.GT.public.ibin` | ~40MB | ibin | Ground truth k-NN results (no filtering) |
| `query.private.*.u8bin` | ~19MB | u8bin | Private test queries (competition) |
| `query.metadata.private.*.spmat` | ~4MB | spmat | Private query metadata |

## File Format Specifications

### u8bin Format (Vectors)

Binary format for dense uint8 vectors.

**Structure**:
```
[Header: 8 bytes]
  - n (uint32): number of vectors
  - d (uint32): dimension

[Data: n × d bytes]
  - Vector data in row-major order (uint8)
```

**Python reader**:
```python
import numpy as np

def read_u8bin(filename):
    """
    Read u8bin format file.

    Args:
        filename: Path to .u8bin file

    Returns:
        np.ndarray of shape (n, d) and dtype uint8
    """
    # Read header
    n, d = np.fromfile(filename, dtype='uint32', count=2)

    # Memory-map the data (efficient for large files)
    vectors = np.memmap(
        filename,
        dtype='uint8',
        mode='r',
        offset=8,  # skip header
        shape=(n, d)
    )

    return vectors

# Example usage
base_vectors = read_u8bin('data/yfcc100M/base.10M.u8bin')
print(f"Shape: {base_vectors.shape}")  # (10000000, 192)
print(f"Dtype: {base_vectors.dtype}")   # uint8
```

**Converting to float32** (REQUIRED for most libraries):
```python
# IMPORTANT: Vectors are uint8, but most libraries require float32
base_vectors_f32 = base_vectors.astype('float32')

print(f"Original dtype: {base_vectors.dtype}")      # uint8
print(f"Converted dtype: {base_vectors_f32.dtype}") # float32
print(f"Value range: [{base_vectors.min()}, {base_vectors.max()}]")  # [0, 255]

# Note: The vectors are already in uint8 range [0, 255]
# FAISS and most libraries will work with this directly after float32 conversion

# Optional: normalize for cosine similarity (not needed for L2)
import sklearn.preprocessing
base_vectors_normalized = sklearn.preprocessing.normalize(
    base_vectors_f32, axis=1, norm='l2'
)
```

**Why uint8?**
- **Space efficiency**: 1.9 GB vs 7.7 GB for float32
- **CLIP quantization**: CLIP embeddings quantized from original float32/float16
- **Competition constraint**: Easier to fit in memory during the competition

### spmat Format (Sparse Metadata)

Binary format for CSR (Compressed Sparse Row) matrices.

**Structure**:
```
[Header: 24 bytes]
  - nrow (int64): number of rows
  - ncol (int64): number of columns
  - nnz (int64): number of non-zero elements

[CSR Data]
  - indptr (int64 × (nrow+1)): row pointers
  - indices (int32 × nnz): column indices
  - data (float32 × nnz): values
```

**Python reader**:
```python
import numpy as np
from scipy.sparse import csr_matrix

def read_spmat(filename):
    """
    Read spmat format file as scipy CSR matrix.

    Args:
        filename: Path to .spmat file

    Returns:
        scipy.sparse.csr_matrix of shape (nrow, ncol)
    """
    with open(filename, 'rb') as f:
        # Read header
        nrow, ncol, nnz = np.fromfile(f, dtype='int64', count=3)

        # Read CSR components
        indptr = np.fromfile(f, dtype='int64', count=nrow + 1)
        assert indptr[-1] == nnz, "nnz mismatch"

        indices = np.fromfile(f, dtype='int32', count=nnz)
        assert np.all(indices >= 0) and np.all(indices < ncol), "invalid indices"

        data = np.fromfile(f, dtype='float32', count=nnz)

    return csr_matrix((data, indices, indptr), shape=(nrow, ncol))

# Example usage
base_metadata = read_spmat('data/yfcc100M/base.metadata.10M.spmat')
print(f"Shape: {base_metadata.shape}")  # (10000000, 200386)
print(f"Non-zeros: {base_metadata.nnz}")  # ~300M
print(f"Sparsity: {100 * base_metadata.nnz / (base_metadata.shape[0] * base_metadata.shape[1]):.4f}%")
```

**Memory-mapped version** (for very large files):
```python
def mmap_spmat(filename):
    """
    Memory-map spmat file for lower memory usage.
    """
    with open(filename, 'rb') as f:
        nrow, ncol, nnz = np.fromfile(f, dtype='int64', count=3)

    # Calculate offsets
    header_size = 3 * 8  # 3 int64s
    indptr_size = (nrow + 1) * 8  # (nrow+1) int64s
    indices_size = nnz * 4  # nnz int32s

    indptr = np.memmap(filename, dtype='int64', mode='r',
                       offset=header_size, shape=nrow + 1)

    indices = np.memmap(filename, dtype='int32', mode='r',
                        offset=header_size + indptr_size, shape=nnz)

    data = np.memmap(filename, dtype='float32', mode='r',
                     offset=header_size + indptr_size + indices_size, shape=nnz)

    return csr_matrix((data, indices, indptr), shape=(nrow, ncol))
```

### ibin Format (Ground Truth)

Binary format for k-NN ground truth results.

**Structure**:
```
[Header: 8 bytes]
  - nq (uint32): number of queries
  - k (uint32): number of neighbors per query

[Indices: nq × k × 4 bytes]
  - Neighbor indices (int32) in row-major order

[Distances: nq × k × 4 bytes]
  - Neighbor distances (float32) in row-major order
```

**Python reader**:
```python
import numpy as np
import os

def read_ibin(filename):
    """
    Read ibin format ground truth file.

    Args:
        filename: Path to .ibin file

    Returns:
        tuple of (I, D) where:
            I: np.ndarray of shape (nq, k), dtype int32 (neighbor indices)
            D: np.ndarray of shape (nq, k), dtype float32 (distances)
    """
    # Read header
    nq, k = np.fromfile(filename, dtype='uint32', count=2)

    # Verify file size
    expected_size = 8 + nq * k * (4 + 4)  # header + indices + distances
    assert os.path.getsize(filename) == expected_size, "File size mismatch"

    with open(filename, 'rb') as f:
        f.seek(8)  # skip header

        # Read neighbor indices
        I = np.fromfile(f, dtype='int32', count=nq * k).reshape(nq, k)

        # Read distances
        D = np.fromfile(f, dtype='float32', count=nq * k).reshape(nq, k)

    return I, D

# Example usage
gt_indices, gt_distances = read_ibin('data/yfcc100M/GT.public.ibin')
print(f"Shape: {gt_indices.shape}")  # (100000, 10)
print(f"Query 0 neighbors: {gt_indices[0]}")
print(f"Query 0 distances: {gt_distances[0]}")
```

## Reading the Data

Complete example to load the entire dataset:

```python
import numpy as np
from scipy.sparse import csr_matrix
import os

class YFCC100MDataset:
    """Reader for YFCC-100M filter track dataset."""

    def __init__(self, data_dir='data/yfcc100M'):
        self.data_dir = data_dir
        self.nb = 10_000_000  # 10M base vectors
        self.nq = 100_000      # 100K queries
        self.d = 192           # dimension
        self.dtype = 'uint8'

    def get_base_vectors(self):
        """Load 10M base vectors."""
        filename = os.path.join(self.data_dir, 'base.10M.u8bin')
        return read_u8bin(filename)

    def get_query_vectors(self):
        """Load 100K query vectors."""
        filename = os.path.join(self.data_dir, 'query.public.100K.u8bin')
        return read_u8bin(filename)

    def get_base_metadata(self):
        """Load base vector metadata (sparse matrix)."""
        filename = os.path.join(self.data_dir, 'base.metadata.10M.spmat')
        return read_spmat(filename)

    def get_query_metadata(self):
        """Load query filter metadata (sparse matrix)."""
        filename = os.path.join(self.data_dir, 'query.metadata.public.100K.spmat')
        return read_spmat(filename)

    def get_groundtruth(self, filtered=True, k=10):
        """Load ground truth k-NN results."""
        if filtered:
            filename = os.path.join(self.data_dir, 'GT.public.ibin')
        else:
            filename = os.path.join(self.data_dir, 'unfiltered.GT.public.ibin')

        I, D = read_ibin(filename)
        return I[:, :k], D[:, :k]

    def distance_metric(self):
        """Distance metric used (L2/Euclidean)."""
        return 'euclidean'

# Usage
dataset = YFCC100MDataset()
base_vecs = dataset.get_base_vectors()
query_vecs = dataset.get_query_vectors()
base_meta = dataset.get_base_metadata()
query_meta = dataset.get_query_metadata()
gt_I, gt_D = dataset.get_groundtruth(filtered=True)
```

## Dataset Characteristics

### Vector Statistics

```python
# Examine the data
print(f"Base vectors: {base_vecs.shape}")      # (10000000, 192)
print(f"Query vectors: {query_vecs.shape}")    # (100000, 192)
print(f"Vector dtype: {base_vecs.dtype}")      # uint8
print(f"Value range: [{base_vecs.min()}, {base_vecs.max()}]")

# Convert to float for analysis
base_f32 = base_vecs.astype('float32')

# Compute norms
norms = np.linalg.norm(base_f32, axis=1)
print(f"Norm statistics:")
print(f"  Mean: {norms.mean():.2f}")
print(f"  Std: {norms.std():.2f}")
print(f"  Min: {norms.min():.2f}")
print(f"  Max: {norms.max():.2f}")
```

### Metadata Statistics

```python
# Analyze metadata sparsity
print(f"Metadata shape: {base_meta.shape}")       # (10000000, 200386)
print(f"Total non-zeros: {base_meta.nnz}")        # ~300M
print(f"Tags per vector (avg): {base_meta.nnz / base_meta.shape[0]:.1f}")
print(f"Sparsity: {100 * base_meta.nnz / base_meta.size:.4f}%")

# Tags per vector distribution
tags_per_vector = np.diff(base_meta.indptr)
print(f"\nTags per vector distribution:")
print(f"  Min: {tags_per_vector.min()}")
print(f"  Max: {tags_per_vector.max()}")
print(f"  Median: {np.median(tags_per_vector):.0f}")
print(f"  Mean: {tags_per_vector.mean():.1f}")

# Query tags distribution
query_tags_per_query = np.diff(query_meta.indptr)
print(f"\nQuery tags distribution:")
print(f"  1 tag: {(query_tags_per_query == 1).sum()}")
print(f"  2 tags: {(query_tags_per_query == 2).sum()}")
print(f"  Other: {(query_tags_per_query > 2).sum()}")
```

## Understanding the Metadata

### Extracting Tags for a Vector

```python
def get_vector_tags(metadata_matrix, vector_id):
    """
    Get the tags (column indices) for a specific vector (row).

    Args:
        metadata_matrix: scipy CSR matrix (vectors × tags)
        vector_id: Row index

    Returns:
        np.ndarray of tag IDs
    """
    row = metadata_matrix.getrow(vector_id)
    return row.indices

# Example: Get tags for base vector 0
tags = get_vector_tags(base_meta, 0)
print(f"Vector 0 has {len(tags)} tags: {tags}")

# Get tags for query 0
query_tags = get_vector_tags(query_meta, 0)
print(f"Query 0 requires tags: {query_tags}")
```

### Building Inverted Index (Tag → Vectors)

```python
def build_inverted_index(metadata_matrix):
    """
    Build inverted index: tag_id → list of vector IDs.

    Args:
        metadata_matrix: CSR matrix (vectors × tags)

    Returns:
        CSR matrix (tags × vectors) - transpose
    """
    # Transpose the matrix
    inverted = metadata_matrix.T.tocsr()
    return inverted

# Build inverted index
inverted_index = build_inverted_index(base_meta)
print(f"Inverted index shape: {inverted_index.shape}")  # (200386, 10000000)

# Find all vectors with tag 100
def get_vectors_with_tag(inverted_index, tag_id):
    """Get all vector IDs that have a specific tag."""
    row = inverted_index.getrow(tag_id)
    return row.indices

vectors_with_tag_100 = get_vectors_with_tag(inverted_index, 100)
print(f"Tag 100 appears in {len(vectors_with_tag_100)} vectors")
```

### Computing Tag Frequencies

```python
def compute_tag_frequencies(metadata_matrix):
    """
    Compute frequency of each tag (fraction of vectors that have it).

    Returns:
        np.ndarray of shape (n_tags,) with frequencies
    """
    inverted = metadata_matrix.T.tocsr()
    tag_counts = np.diff(inverted.indptr)
    frequencies = tag_counts / metadata_matrix.shape[0]
    return frequencies

tag_freqs = compute_tag_frequencies(base_meta)
print(f"Tag frequency statistics:")
print(f"  Min: {tag_freqs.min():.6f}")
print(f"  Max: {tag_freqs.max():.6f}")
print(f"  Median: {np.median(tag_freqs):.6f}")

# Find rare vs common tags
rare_tags = np.where(tag_freqs < 0.001)[0]  # < 0.1%
common_tags = np.where(tag_freqs > 0.01)[0]  # > 1%
print(f"Rare tags (< 0.1%): {len(rare_tags)}")
print(f"Common tags (> 1%): {len(common_tags)}")
```

## Implementing Filtered Search

### Strategy 1: Post-Filtering

Retrieve candidates first, then filter by metadata.

**Pros**: Simple to implement with any vector search library
**Cons**: Inefficient, may need to retrieve many candidates

```python
def filtered_search_postfilter(index, base_meta, query_vec, query_tags, k,
                                 retrieval_factor=10):
    """
    Post-filtering approach: retrieve candidates then filter.

    Args:
        index: Your vector search index
        base_meta: Base metadata CSR matrix
        query_vec: Query vector
        query_tags: Required tag IDs (array)
        k: Number of results to return
        retrieval_factor: How many candidates to retrieve (k * factor)

    Returns:
        np.ndarray of k neighbor IDs
    """
    # Retrieve more candidates than needed
    n_candidates = k * retrieval_factor
    candidate_ids = index.search(query_vec, k=n_candidates)

    # Filter candidates by tags
    results = []
    required_tags = set(query_tags)

    for cand_id in candidate_ids:
        # Get tags for this candidate
        cand_tags = set(base_meta.getrow(cand_id).indices)

        # Check if all required tags are present
        if required_tags.issubset(cand_tags):
            results.append(cand_id)
            if len(results) >= k:
                break

    # Pad with -1 if not enough results
    while len(results) < k:
        results.append(-1)

    return np.array(results[:k])

# Example usage
# Assuming you have a vector search index built
# results = filtered_search_postfilter(
#     my_index, base_meta, query_vecs[0], query_tags[0], k=10
# )
```

### Strategy 2: Pre-Filtering

Filter first using inverted index, then search among valid vectors.

**Pros**: Efficient for rare tags
**Cons**: Doesn't scale for common tags (too many candidates)

```python
def filtered_search_prefilter(base_vecs, base_meta, query_vec, query_tags, k):
    """
    Pre-filtering approach: filter first, then compute k-NN.

    Args:
        base_vecs: Base vectors array
        base_meta: Base metadata CSR matrix
        query_vec: Query vector
        query_tags: Required tag IDs (array)
        k: Number of results to return

    Returns:
        np.ndarray of k neighbor IDs
    """
    # Build inverted index (can be pre-computed)
    inverted = base_meta.T.tocsr()

    # Get vectors matching first tag
    valid_ids = set(inverted.getrow(query_tags[0]).indices)

    # Intersect with other tags
    for tag in query_tags[1:]:
        tag_vectors = set(inverted.getrow(tag).indices)
        valid_ids &= tag_vectors

    valid_ids = np.array(list(valid_ids))

    if len(valid_ids) < k:
        # Not enough results
        result = np.full(k, -1, dtype='int32')
        result[:len(valid_ids)] = valid_ids
        return result

    # Compute distances only for valid vectors
    valid_vecs = base_vecs[valid_ids].astype('float32')
    query_vec_f32 = query_vec.astype('float32')

    distances = np.linalg.norm(valid_vecs - query_vec_f32, axis=1)

    # Get k nearest
    top_k_indices = np.argpartition(distances, min(k, len(distances) - 1))[:k]
    top_k_indices = top_k_indices[np.argsort(distances[top_k_indices])]

    return valid_ids[top_k_indices]

# Example usage
results = filtered_search_prefilter(
    base_vecs, base_meta, query_vecs[0],
    get_vector_tags(query_meta, 0), k=10
)
print(f"Results: {results}")
```

### Strategy 3: Hybrid Approach (FAISS Baseline)

Use different strategies based on tag frequency.

```python
def filtered_search_hybrid(base_vecs, base_meta, query_vec, query_tags, k,
                            threshold=0.001):
    """
    Hybrid approach: choose strategy based on tag frequency.

    Args:
        base_vecs: Base vectors array
        base_meta: Base metadata CSR matrix
        query_vec: Query vector
        query_tags: Required tag IDs (array)
        k: Number of results to return
        threshold: Frequency threshold for strategy selection

    Returns:
        np.ndarray of k neighbor IDs
    """
    # Compute tag frequencies
    inverted = base_meta.T.tocsr()
    tag_freqs = np.diff(inverted.indptr) / base_meta.shape[0]

    # Get frequency of query tags
    query_freq = tag_freqs[query_tags[0]]
    for tag in query_tags[1:]:
        query_freq *= tag_freqs[tag]  # Joint frequency (approximate)

    if query_freq < threshold:
        # Rare tags: use pre-filtering (metadata-first)
        print(f"Using pre-filtering (freq={query_freq:.6f})")
        return filtered_search_prefilter(
            base_vecs, base_meta, query_vec, query_tags, k
        )
    else:
        # Common tags: use post-filtering (IVF-first)
        # Or use filtered IVF search if your library supports it
        print(f"Using post-filtering (freq={query_freq:.6f})")
        # This would use your IVF index with filtering
        # For now, fall back to pre-filtering for demonstration
        return filtered_search_prefilter(
            base_vecs, base_meta, query_vec, query_tags, k
        )

# Example usage
results = filtered_search_hybrid(
    base_vecs, base_meta, query_vecs[0],
    get_vector_tags(query_meta, 0), k=10
)
```

### Optimized: Sorted Intersection for Multiple Tags

```python
def intersect_sorted(arr1, arr2):
    """
    Fast intersection of two sorted arrays.

    Args:
        arr1, arr2: Sorted numpy arrays

    Returns:
        np.ndarray of intersection
    """
    i, j = 0, 0
    result = []

    while i < len(arr1) and j < len(arr2):
        if arr1[i] < arr2[j]:
            i += 1
        elif arr1[i] > arr2[j]:
            j += 1
        else:
            result.append(arr1[i])
            i += 1
            j += 1

    return np.array(result, dtype=arr1.dtype)

def get_filtered_candidates(base_meta, query_tags):
    """
    Efficiently find all vectors matching all query tags.

    Args:
        base_meta: Base metadata CSR matrix
        query_tags: Array of required tag IDs

    Returns:
        np.ndarray of matching vector IDs
    """
    inverted = base_meta.T.tocsr()

    # Get vectors for first tag (already sorted)
    candidates = inverted.getrow(query_tags[0]).indices.copy()

    # Intersect with remaining tags
    for tag in query_tags[1:]:
        tag_vecs = inverted.getrow(tag).indices
        candidates = intersect_sorted(candidates, tag_vecs)

        if len(candidates) == 0:
            break

    return candidates
```

## Evaluation

### Computing Recall

```python
def compute_recall(retrieved, groundtruth, k=10):
    """
    Compute recall@k for retrieved results.

    Args:
        retrieved: Array of retrieved neighbor IDs (nq, k)
        groundtruth: Array of ground truth neighbor IDs (nq, k)
        k: Number of neighbors to consider

    Returns:
        float: Average recall across all queries
    """
    nq = len(groundtruth)
    recalls = []

    for q in range(nq):
        gt_set = set(groundtruth[q, :k])
        ret_set = set(retrieved[q, :k])

        # Remove -1 (padding)
        gt_set.discard(-1)
        ret_set.discard(-1)

        if len(gt_set) == 0:
            continue

        recall = len(gt_set & ret_set) / len(gt_set)
        recalls.append(recall)

    return np.mean(recalls)

# Example usage
gt_I, _ = dataset.get_groundtruth(filtered=True, k=10)

# Run your search algorithm
my_results = np.zeros((100000, 10), dtype='int32')
for q in range(100000):
    query_vec = query_vecs[q]
    query_tags = get_vector_tags(query_meta, q)
    my_results[q] = filtered_search_prefilter(
        base_vecs, base_meta, query_vec, query_tags, k=10
    )

# Compute recall
recall = compute_recall(my_results, gt_I, k=10)
print(f"Recall@10: {recall:.4f}")
```

### Computing QPS (Queries Per Second)

```python
import time

def benchmark_search(search_fn, query_vecs, query_meta, k=10, n_queries=1000):
    """
    Benchmark search performance.

    Args:
        search_fn: Function(query_vec, query_tags, k) -> results
        query_vecs: Query vectors
        query_meta: Query metadata
        k: Number of neighbors
        n_queries: Number of queries to benchmark

    Returns:
        dict with performance metrics
    """
    start_time = time.time()

    for q in range(n_queries):
        query_vec = query_vecs[q]
        query_tags = get_vector_tags(query_meta, q)
        _ = search_fn(query_vec, query_tags, k)

    elapsed = time.time() - start_time
    qps = n_queries / elapsed
    latency = elapsed / n_queries * 1000  # ms

    return {
        'qps': qps,
        'latency_ms': latency,
        'total_time_s': elapsed
    }

# Example
def my_search_fn(query_vec, query_tags, k):
    return filtered_search_prefilter(
        base_vecs, base_meta, query_vec, query_tags, k
    )

metrics = benchmark_search(my_search_fn, query_vecs, query_meta, k=10, n_queries=100)
print(f"QPS: {metrics['qps']:.1f}")
print(f"Latency: {metrics['latency_ms']:.2f} ms")
```

## Complete Example

Full working example with all components:

```python
import numpy as np
from scipy.sparse import csr_matrix
import time

# 1. Load dataset
print("Loading dataset...")
dataset = YFCC100MDataset('data/yfcc100M')
base_vecs = dataset.get_base_vectors()
query_vecs = dataset.get_query_vectors()
base_meta = dataset.get_base_metadata()
query_meta = dataset.get_query_metadata()
gt_I, gt_D = dataset.get_groundtruth(filtered=True, k=10)

print(f"Loaded {base_vecs.shape[0]} base vectors")
print(f"Loaded {query_vecs.shape[0]} query vectors")
print(f"Metadata: {base_meta.shape}")

# 2. Build inverted index (pre-compute)
print("Building inverted index...")
inverted_index = base_meta.T.tocsr()

# 3. Run filtered search on test queries
print("Running filtered search...")
k = 10
n_test = 1000  # Test on first 1000 queries

results = np.zeros((n_test, k), dtype='int32')
start_time = time.time()

for q in range(n_test):
    query_vec = query_vecs[q]
    query_tags = query_meta.getrow(q).indices

    # Use pre-filtering approach
    results[q] = filtered_search_prefilter(
        base_vecs, base_meta, query_vec, query_tags, k
    )

    if (q + 1) % 100 == 0:
        print(f"  Processed {q + 1}/{n_test} queries...")

elapsed = time.time() - start_time

# 4. Evaluate
recall = compute_recall(results, gt_I[:n_test], k=10)
qps = n_test / elapsed

print(f"\nResults:")
print(f"  Recall@10: {recall:.4f}")
print(f"  QPS: {qps:.1f}")
print(f"  Avg latency: {1000/qps:.2f} ms")

# 5. Analyze some queries
print(f"\nExample queries:")
for q in range(3):
    query_tags = query_meta.getrow(q).indices
    print(f"Query {q}:")
    print(f"  Required tags: {query_tags}")
    print(f"  Retrieved: {results[q]}")
    print(f"  Groundtruth: {gt_I[q]}")
    print(f"  Match: {len(set(results[q]) & set(gt_I[q]))}/{k}")
```

## Performance Considerations

### Memory Usage

```python
# Estimate memory requirements
base_vecs_mb = base_vecs.nbytes / 1024**2
base_meta_mb = (base_meta.data.nbytes + base_meta.indices.nbytes +
                base_meta.indptr.nbytes) / 1024**2

print(f"Memory usage:")
print(f"  Base vectors: {base_vecs_mb:.1f} MB")
print(f"  Base metadata: {base_meta_mb:.1f} MB")
print(f"  Total: {base_vecs_mb + base_meta_mb:.1f} MB")
```

### Optimization Tips

1. **Use memory mapping** for vectors:
   ```python
   # Instead of loading into RAM
   base_vecs = np.memmap('data/yfcc100M/base.10M.u8bin',
                         dtype='uint8', mode='r', offset=8,
                         shape=(10000000, 192))
   ```

2. **Pre-compute inverted index** and cache:
   ```python
   import pickle

   # Build once
   inverted = base_meta.T.tocsr()
   with open('inverted_index.pkl', 'wb') as f:
       pickle.dump(inverted, f)

   # Load later
   with open('inverted_index.pkl', 'rb') as f:
       inverted = pickle.load(f)
   ```

3. **Batch queries** for better throughput:
   ```python
   def batch_filtered_search(base_vecs, base_meta, query_vecs_batch,
                              query_tags_batch, k):
       """Process multiple queries together."""
       results = []
       for query_vec, query_tags in zip(query_vecs_batch, query_tags_batch):
           result = filtered_search_prefilter(
               base_vecs, base_meta, query_vec, query_tags, k
           )
           results.append(result)
       return np.array(results)
   ```

4. **Parallelize** query processing:
   ```python
   from multiprocessing import Pool

   def process_query(args):
       q_idx, query_vec, query_tags = args
       return filtered_search_prefilter(
           base_vecs, base_meta, query_vec, query_tags, k=10
       )

   with Pool(8) as pool:
       query_args = [(q, query_vecs[q], get_vector_tags(query_meta, q))
                     for q in range(1000)]
       results = pool.map(process_query, query_args)
   ```

### FAISS Baseline Performance

Reference performance from NeurIPS 2023 competition:

- **Platform**: Azure Standard D8lds v5 (8 vCPUs, 16GB RAM)
- **CPU**: Intel Xeon Platinum 8370C @ 2.80GHz
- **Metric**: QPS at 90% recall
- **Result**: 3200 QPS

Configuration used:
- Index: `IVF16384,SQ8` with binary signatures
- Query params: Variable `nprobe` and `mt_threshold`
- Strategy: Hybrid (metadata-first for rare tags, IVF-first for common)

---

For more information, see:
- [NeurIPS 2023 Competition README](neurips23/README.md)
- [FAISS baseline implementation](neurips23/filter/faiss/faiss.py)
- [Dataset downloads](https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/yfcc100M/)
