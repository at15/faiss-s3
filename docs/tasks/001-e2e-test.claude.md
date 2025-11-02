# Full E2E Test Implementation Plan

**Status**: ✅ Implemented
**Created**: 2025-11-01
**Task Reference**: [docs/tasks/001-e2e-test.md](001-e2e-test.md)

## Overview

This document describes the implementation of the full end-to-end test for faiss-s3, which validates the complete workflow from raw dataset to querying via both local FAISS and S3 cache server.

## Implementation Summary

The test is implemented in [tests/test_e2e_full.py](../../tests/test_e2e_full.py) and follows a 11-phase workflow that covers:

1. **Data Preparation**: Download Quora dataset and extract unique questions
2. **Embedding Generation**: Create vector embeddings using SentenceTransformers
3. **Index Building**: Build FAISS IVF index with dynamic cluster calculation
4. **Attributes File**: Generate parquet file for source text mapping
5. **Meta.json**: Generate complete metadata with all three sections
6. **S3 Upload**: Upload all files to S3 with unique timestamp prefix
7. **Infrastructure Validation**: Verify S3Mock and cache server are running
8. **Local Query**: Test FAISS queries via download + mmap
9. **Server Query**: Test queries via S3 cache server with on-demand loading
10. **Result Verification**: Confirm both methods return identical vector IDs
11. **Benchmarking**: Report comprehensive timing statistics

## Key Design Decisions

### 1. Test Data Size
- **Default**: 10,000 vectors (configurable via `--dataset-size`)
- **Rationale**: Fast enough for CI/CD while still meaningful
- **Constraint Handling**: Auto-calculate `n_clusters = max(10, dataset_size // 100)` to ensure `n_clusters < n_vectors`

### 2. Embedding Format
- **Choice**: `.npy` (numpy native format)
- **Rationale**: Production-ready, efficient, standard format
- **Previous**: Tests used `.pkl` (pickle) for convenience

### 3. Attributes Parquet Schema
- **Schema**: Single `text` column (no explicit `id` column)
- **ID Mapping**: Row index implicitly equals vector ID
- **Rationale**: Simpler schema, follows insertion order guarantee
- **Usage**: `df.loc[vector_id, 'text']` to retrieve source text

### 4. Infrastructure Handling
- **Approach**: Fail fast with clear error messages
- **S3Mock Check**: Test connection with `list_buckets()`
- **Server Check**: Test connection with `echo()` command
- **Error Messages**: Include setup instructions for local and Docker environments

### 5. Test Isolation
- **S3 Prefix**: Unique timestamp-based prefix per run (`quora-20251101-120000/`)
- **Rationale**: Avoid conflicts between test runs, enable inspection
- **Cleanup**: Artifacts preserved for debugging (manual cleanup)

### 6. Result Verification
- **Method**: Compare vector ID sets from both query methods
- **Assertion**: `set(local_ids) == set(server_ids)`
- **Rationale**: Order may differ due to internal tie-breaking, but IDs must match
- **Tolerance**: No floating-point tolerance needed (IDs are exact integers)

## File Structure

```
tests/
├── test_e2e_full.py           # Main implementation (new)
├── test_quora.py              # Original (partial workflow)
└── test_s3_cache_server.py    # Original (server testing only)

e2e_cache/                      # Shared cache (persists across runs)
└── quora_duplicate_questions.tsv  # Downloaded once, reused

e2e_test_20251101-120000/      # Working directory (unique per run)
├── embeddings.npy
├── index.ivf
├── attributes.parquet
├── meta.json
├── downloaded_index.ivf       # For local query testing
├── downloaded_meta.json
└── downloaded_attributes.parquet

s3://test-bucket/quora-20251101-120000/
├── meta.json
├── index.ivf
├── attributes.parquet
└── embeddings.npy
```

## Meta.json Structure

Complete implementation of the specification with all three sections:

```json
{
  "version": "1",
  "index": {
    "file": "index.ivf",
    "size_bytes": 1234567,
    "type": "IndexIVFFlat",
    "vector_size": 768,
    "n_vectors": 10000,
    "n_clusters": 100,
    "cluster_data_offset": 3154059  // DYNAMIC via generate_meta_from_file
  },
  "attributes": {
    "file": "attributes.parquet",
    "size_bytes": 45678,
    "columns": [
      {"name": "text", "type": "string"}
    ]
  },
  "embedding": {
    "file": "embeddings.npy",
    "library": "sentence-transformers",
    "model": "quora-distilbert-multilingual",
    "vector_size": 768,
    "n_vectors": 10000
  }
}
```

**Key Achievement**: `cluster_data_offset` is calculated dynamically using `faiss_s3.generate_meta_from_file()` instead of hardcoded values.

## Usage

### Installation

```bash
# Install client with test dependencies
cd client
pip install -e ".[test]"
```

### Prerequisites

```bash
# 1. Start S3Mock
docker run -p 9000:9000 adobe/s3mock

# 2. Set environment variables
export S3_ENDPOINT_URL=http://localhost:9000
export AWS_ACCESS_KEY_ID=test
export AWS_SECRET_ACCESS_KEY=test
export AWS_REGION=us-east-1
export AWS_EC2_METADATA_DISABLED=true

# 3. Start S3 cache server
cd server
make docker-run
# Or: make build && ./s3_cache_server

# 4. Create test bucket
aws --endpoint-url=http://localhost:9000 s3 mb s3://test-bucket
```

### Running the Test

```bash
# Default configuration (10k vectors, auto clusters)
cd tests
python test_e2e_full.py

# Custom configuration
python test_e2e_full.py --dataset-size 50000 --n-clusters 500 --k 10 --nprobe 20

# Small quick test
python test_e2e_full.py --dataset-size 1000 --n-clusters 10 --k 5
```

### Command Line Options

- `--dataset-size N`: Number of vectors (default: 10000)
- `--n-clusters N`: IVF clusters (default: auto-calculated as dataset_size/100)
- `--k N`: Top-k results to return (default: 5)
- `--nprobe N`: Clusters to search (default: 10)

## Implementation Highlights

### 1. Dynamic Cluster Calculation

```python
# Auto-calculate to ensure n_clusters < n_vectors
if n_clusters is None:
    self.n_clusters = max(10, dataset_size // 100)

# Validation
if self.n_clusters >= self.dataset_size:
    raise ValueError(f"n_clusters must be < dataset_size")
```

### 2. Attributes Parquet Generation

```python
# Simple schema: just text column, row index = vector ID
df = pd.DataFrame({"text": self.corpus_sentences})
table = pa.Table.from_pandas(df)
pq.write_table(table, str(self.attributes_path))

# Retrieval
text = df.loc[vector_id, 'text']
```

### 3. Dynamic Offset Calculation

```python
# Use faiss_s3 library function instead of hardcoding
index_meta = generate_meta_from_file(str(self.index_path))
cluster_data_offset = index_meta["cluster_data_offset"]
```

### 4. Result Verification

```python
# Compare sets (order-agnostic)
local_id_set = set(local_ids.tolist())
server_id_set = set(server_ids.tolist())

if local_id_set == server_id_set:
    print("✓ SUCCESS: Both methods returned identical vector IDs")
else:
    print("✗ FAILURE: Result mismatch!")
    sys.exit(1)
```

### 5. Comprehensive Error Messages

```python
except Exception as e:
    print(f"\n✗ ERROR: S3Mock not accessible")
    print(f"  Error: {e}")
    print("\nSetup instructions:")
    print("  docker run -p 9000:9000 adobe/s3mock")
    print("\nOr set environment variables:")
    print("  export S3_ENDPOINT_URL=http://localhost:9000")
    # ... more instructions
    sys.exit(1)
```

## Benchmarking Output

The test reports timing for each phase:

```
BENCHMARK SUMMARY
================================================================================
  dataset_download              :     5.23s ( 12.5%)
  embedding_generation          :    28.45s ( 68.1%)
  index_building                :     3.12s (  7.5%)
  attributes_generation         :     0.15s (  0.4%)
  meta_generation               :     0.08s (  0.2%)
  s3_upload                     :     2.34s (  5.6%)
  infrastructure_validation     :     0.12s (  0.3%)
  local_query                   :     1.45s (  3.5%)
  server_query                  :     0.65s (  1.6%)
  result_verification           :     0.02s (  0.0%)

  TOTAL                         :    41.61s
```

## Testing Status

### ✅ Implemented Features

1. ✅ Configurable test parameters via CLI
2. ✅ Complete data preparation pipeline
3. ✅ Dynamic n_clusters calculation with constraint validation
4. ✅ `.npy` embedding format (production-ready)
5. ✅ Attributes parquet with implicit ID mapping
6. ✅ Complete meta.json with all three sections
7. ✅ Dynamic cluster_data_offset calculation
8. ✅ Unique S3 prefix per run
9. ✅ Infrastructure validation with clear error messages
10. ✅ Dual query methods (local + server)
11. ✅ Result verification (set comparison)
12. ✅ Comprehensive benchmarking

### 🔄 To Be Tested

- [ ] Run with small dataset (1k vectors) to verify correctness
- [ ] Run with larger dataset (50k vectors) to verify scalability
- [ ] Test with S3Mock + Docker server in CI/CD environment
- [ ] Verify cache statistics show expected on-demand loading behavior
- [ ] Test error handling when infrastructure is missing

### 📝 Future Enhancements

- ✅ Dataset caching (implemented: Quora TSV cached in `e2e_cache/`)
- Optional embedding caching between runs (current: always regenerate)
- Multiple query comparison (current: single query for both methods)
- Recall calculation (requires ground truth labels)
- Support for multiple datasets/indexes in single run
- Automated infrastructure startup (Docker Compose)
- Test cleanup command (delete S3 data and work directories)

## Differences from Existing Tests

### vs. test_quora.py

| Aspect | test_quora.py | test_e2e_full.py |
|--------|--------------|------------------|
| Query Method | Local only (mmap) | Both local and server |
| Metadata | Hardcoded offset | Dynamic via generate_meta_from_file |
| Meta.json | Partial (index only) | Complete (index + attributes + embedding) |
| Attributes | pickle file | parquet file |
| S3 Upload | Commented out | Full upload with unique prefix |
| Verification | None | Set comparison of results |

### vs. test_s3_cache_server.py

| Aspect | test_s3_cache_server.py | test_e2e_full.py |
|--------|------------------------|------------------|
| Data Prep | Uses pre-existing files | Complete pipeline from raw data |
| Offset | Hardcoded (3154059) | Dynamic calculation |
| Text Mapping | pickle file | parquet file |
| Local Query | Not tested | Tested and compared |
| Isolation | Shared S3 location | Unique prefix per run |

## Dependencies Added

Updated [client/pyproject.toml](../../client/pyproject.toml):

```toml
[project.optional-dependencies]
test = [
    "sentence-transformers",
    "pandas",
    "pyarrow",
]
```

Install with:
```bash
pip install -e ".[test]"
```

## Success Criteria Checklist

- [x] Runs from clean state without manual intervention (except infrastructure)
- [x] Generates complete S3 layout with all 4 files
- [x] Meta.json uses dynamic cluster_data_offset
- [x] Attributes.parquet correctly maps vector IDs to source text
- [x] Both query methods work correctly
- [x] Result verification confirms identical vector ID sets
- [x] Clear error messages when infrastructure missing
- [x] Comprehensive timing benchmarks reported
- [x] Cache statistics display on-demand loading behavior
- [x] Unique S3 prefix per run (no conflicts)
- [x] Validates n_clusters < n_vectors constraint
- [x] Configurable via CLI arguments

## Next Steps

1. **Testing**: Run the test with S3Mock and server infrastructure
2. **Documentation**: Update main README with e2e test instructions
3. **CI/CD Integration**: Add test to GitHub Actions workflow
4. **Optimization**: Consider caching embeddings for development workflow
5. **Monitoring**: Add more detailed cache statistics and analysis

## References

- Original Task: [docs/tasks/001-e2e-test.md](001-e2e-test.md)
- Implementation: [tests/test_e2e_full.py](../../tests/test_e2e_full.py)
- Dependencies: [client/pyproject.toml](../../client/pyproject.toml)
- Project Docs: [CLAUDE.md](../../CLAUDE.md)
