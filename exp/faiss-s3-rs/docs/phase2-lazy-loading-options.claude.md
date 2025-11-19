---
tags:
  - rust
  - faiss
  - s3
  - architecture
---

# Phase 2: Lazy Cluster Loading Options

## Problem Statement

When implementing lazy cluster data loading, we face a fundamental async/blocking mismatch:

```
┌─────────────┐
│ Async Rust  │ (tokio runtime)
│ search()    │
└──────┬──────┘
       │ calls C++ IndexIVF::search()
       ↓
┌──────────────┐
│ C++ (sync)   │ Blocking, expects immediate return
│ IndexIVF     │
│ ::search()   │
└──────┬───────┘
       │ calls get_codes(list_no)
       ↓
┌────────────────────┐
│ S3RustIO...Lists   │ (C++ wrapper)
│ fetch_codes_cb()   │
└──────┬─────────────┘
       │ callback to Rust
       ↓
┌─────────────────┐
│ Rust S3 fetch   │ ⚠️ PROBLEM: Need async here!
│ s3.get_range()  │    But called from sync C++
└─────────────────┘
```

### Key Constraints

1. **C++ is blocking**: Faiss search expects synchronous `get_codes()` that returns immediately
2. **object_store is async-only**: We MUST use `object_store` for multi-backend support (S3, GCS, Azure, local, etc.)
3. **No blocking HTTP alternative**: Cannot use `reqwest::blocking` or `ureq` as it defeats the purpose of `object_store`
4. **Tokio HTTP server context**: Later this will run in a tokio-based HTTP server where all requests share the same runtime
5. **Deadlock risk**: **CRITICAL** - Using `Handle::block_on()` from within a tokio runtime task will deadlock when all workers are blocked
6. **Callback context**: C++ doesn't understand Rust async/await

## Option 1: Dedicated Runtime per Fetch (Simplest)

### Approach

Create a new tokio runtime for each S3 fetch request and block on it.

### Implementation

```rust
use std::sync::Arc;
use object_store::ObjectStore;

struct ClusterFetcher {
    s3: Arc<dyn ObjectStore>,
    bucket: String,
    index_path: String,
    cluster_data_offset: usize,
}

impl ClusterFetcher {
    fn fetch_cluster_sync(&self, list_no: usize, offset: usize, size: usize) -> Vec<u8> {
        // Create a new runtime for this fetch
        let rt = tokio::runtime::Runtime::new().unwrap();

        rt.block_on(async {
            let path = object_store::path::Path::from(self.index_path.clone());
            let range = (self.cluster_data_offset + offset)
                ..(self.cluster_data_offset + offset + size);

            self.s3.get_range(&path, range)
                .await
                .unwrap()
                .to_vec()
        })
    }
}

// C++ callback uses this
extern "C" fn fetch_codes_callback(
    context: *mut std::ffi::c_void,
    list_no: usize,
) -> *const u8 {
    let fetcher = unsafe { &*(context as *const ClusterFetcher) };
    // Calculate offset and size based on list_no
    let data = fetcher.fetch_cluster_sync(list_no, offset, size);
    // Store and return pointer (need proper lifetime management)
    Box::leak(data.into_boxed_slice()).as_ptr()
}
```

### Pros

- Simple to implement
- Works with existing async S3 code
- No shared state or channels needed
- Each fetch is independent

### Cons

- **Performance overhead**: Creating a runtime per fetch is expensive
- **Resource usage**: Each runtime spawns threads
- Not suitable for high-frequency fetches
- Runtime creation can fail

### Best For

- Prototyping
- Low-frequency fetches
- Testing the overall architecture

## Option 2: Shared Runtime with `block_on`

### Approach

Use a single shared tokio runtime, but call `Handle::block_on()` from callbacks.

### Implementation

```rust
use tokio::runtime::Handle;
use once_cell::sync::Lazy;

static RUNTIME: Lazy<tokio::runtime::Runtime> = Lazy::new(|| {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4) // Dedicated for S3 fetches
        .thread_name("s3-fetch")
        .enable_all()
        .build()
        .unwrap()
});

struct ClusterFetcher {
    s3: Arc<dyn ObjectStore>,
    // ... other fields
}

impl ClusterFetcher {
    fn fetch_cluster_sync(&self, list_no: usize) -> Vec<u8> {
        let handle = RUNTIME.handle();

        // Block this thread while waiting for async operation
        handle.block_on(async {
            self.s3.get_range(&path, range).await.unwrap().to_vec()
        })
    }
}
```

### Pros

- Single runtime reduces overhead
- Reuses thread pool
- Still works with async S3 libraries

### Cons

- **Deadlock risk**: If C++ is called from tokio thread, `block_on` can deadlock
- Requires careful thread separation
- Still blocks the calling thread

### Best For

- Medium-frequency fetches
- When C++ search is always called from non-tokio threads
- Better performance than Option 1

## Option 3: Channel-Based Async Bridge

### Approach

Use channels to send fetch requests from sync callbacks to an async task.

### Implementation

```rust
use tokio::sync::mpsc;
use tokio::sync::oneshot;

struct FetchRequest {
    list_no: usize,
    offset: usize,
    size: usize,
    reply_tx: oneshot::Sender<Vec<u8>>,
}

struct ClusterFetcher {
    fetch_tx: mpsc::Sender<FetchRequest>,
}

impl ClusterFetcher {
    fn new(s3: Arc<dyn ObjectStore>) -> Self {
        let (tx, mut rx) = mpsc::channel::<FetchRequest>(100);

        // Spawn background task to process fetch requests
        tokio::spawn(async move {
            while let Some(req) = rx.recv().await {
                let data = s3.get_range(&path, req.offset..(req.offset + req.size))
                    .await
                    .unwrap()
                    .to_vec();

                let _ = req.reply_tx.send(data);
            }
        });

        Self { fetch_tx: tx }
    }

    fn fetch_cluster_sync(&self, list_no: usize, offset: usize, size: usize) -> Vec<u8> {
        let (reply_tx, reply_rx) = oneshot::channel();

        // Send request to async task
        self.fetch_tx.blocking_send(FetchRequest {
            list_no,
            offset,
            size,
            reply_tx,
        }).unwrap();

        // Block waiting for response
        reply_rx.blocking_recv().unwrap()
    }
}
```

### Pros

- Separates async work from sync callbacks
- Single tokio runtime in background
- Can batch or optimize requests in async task
- Better for high-frequency requests

### Cons

- More complex implementation
- Still blocks the calling thread
- Channel overhead
- Need proper shutdown handling

### Best For

- High-frequency fetches
- When you want to optimize/batch requests
- Production systems with predictable load

## Option 4: Separate Thread Pool with `block_on`

### Approach

Create a dedicated thread pool for C++ search operations. Each thread has its own tokio runtime that can safely call `block_on` without deadlocking the main runtime.

### Implementation

```rust
use std::sync::Arc;
use object_store::ObjectStore;
use tokio::runtime::Runtime;
use threadpool::ThreadPool;

struct ClusterFetcher {
    s3: Arc<dyn ObjectStore>,
    index_path: String,
    cluster_data_offset: usize,
    // Each thread in this pool has its own tokio runtime
    search_thread_pool: Arc<ThreadPool>,
}

impl ClusterFetcher {
    fn new(
        s3: Arc<dyn ObjectStore>,
        index_path: String,
        cluster_data_offset: usize,
    ) -> Self {
        // Create thread pool for search operations
        // Each thread will create its own tokio runtime
        let search_thread_pool = Arc::new(ThreadPool::new(4));

        Self {
            s3,
            index_path,
            cluster_data_offset,
            search_thread_pool,
        }
    }

    fn fetch_cluster_sync(&self, offset: usize, size: usize) -> Vec<u8> {
        // Create a runtime in this thread (thread-local)
        thread_local! {
            static RT: Runtime = Runtime::new().unwrap();
        }

        RT.with(|rt| {
            rt.block_on(async {
                let path = object_store::path::Path::from(self.index_path.clone());
                let range = (self.cluster_data_offset + offset)
                    ..(self.cluster_data_offset + offset + size);

                self.s3.get_range(&path, range)
                    .await
                    .unwrap()
                    .to_vec()
            })
        })
    }
}

// C++ search must be called from the dedicated thread pool
impl ClusterFetcher {
    fn search_in_thread_pool(&self, query: Vec<f32>, k: usize) -> Vec<(i64, f32)> {
        let (tx, rx) = std::sync::mpsc::channel();
        let fetcher = self.clone();

        self.search_thread_pool.execute(move || {
            // This runs in a dedicated thread, safe to block_on
            let result = perform_cpp_search(&fetcher, query, k);
            tx.send(result).unwrap();
        });

        rx.recv().unwrap()
    }
}
```

### Pros

- Uses `object_store` (maintains multi-backend support)
- Dedicated thread pool prevents deadlock with main runtime
- Thread-local runtimes are reused
- Clean separation between async and sync worlds

### Cons

- Thread pool overhead
- Need to ensure C++ search only runs in thread pool
- Each thread has its own runtime (memory overhead)
- Complexity in managing thread-local runtimes

### Best For

- When you must use `object_store`
- Medium-frequency searches
- When you can control where C++ search is called from

## Option 5: Make Search Itself Async (Recommended for Server)

### Approach

**KEY INSIGHT**: Instead of trying to call async from sync C++, make the entire search operation async from the Rust side!

**Architecture**:
1. Rust exposes `async fn search()` to the HTTP server
2. Inside async search, do synchronous C++ calls for quantizer
3. Await parallel cluster prefetch
4. Pass prefetched data to C++ for final search

This avoids `block_on` entirely - everything is properly async.

### Implementation

```rust
use std::sync::Arc;
use std::collections::HashMap;
use tokio::sync::RwLock;

pub struct FaissIndexS3 {
    inner: Arc<FaissIVFIndexS3Wrapper>,
    cluster_fetcher: Arc<ClusterFetcher>,
}

struct ClusterFetcher {
    s3: Arc<dyn ObjectStore>,
    index_path: String,
    cluster_data_offset: usize,
    cluster_metadata: Vec<ClusterMeta>,
    // Store prefetched data per-search (thread-safe)
    prefetch_cache: Arc<RwLock<HashMap<usize, Vec<u8>>>>,
}

impl FaissIndexS3 {
    /// Async search - callable from tokio HTTP server
    pub async fn search(
        &self,
        query: &[f32],
        nprobe: usize,
        k: usize,
    ) -> Result<SearchResults, Error> {
        // Step 1: Call C++ quantizer (sync, fast)
        let cluster_ids = self.inner.quantizer_search(query, nprobe);

        // Step 2: Prefetch clusters in parallel (async, slow)
        let cluster_data = self.cluster_fetcher
            .prefetch_clusters_parallel(&cluster_ids)
            .await?;

        // Step 3: Store prefetched data for C++ to access
        {
            let mut cache = self.prefetch_cache.write().await;
            cache.clear();
            cache.extend(cluster_data);
        }

        // Step 4: Call C++ search_preassigned (sync, uses prefetched data)
        let results = self.inner.search_preassigned(query, k, &cluster_ids);

        Ok(results)
    }
}

impl ClusterFetcher {
    /// Prefetch clusters in parallel - fully async, no block_on
    async fn prefetch_clusters_parallel(
        &self,
        cluster_ids: &[usize],
    ) -> Result<HashMap<usize, Vec<u8>>, Error> {
        let mut futures = Vec::new();

        for &cluster_id in cluster_ids {
            let s3 = self.s3.clone();
            let index_path = self.index_path.clone();
            let meta = self.cluster_metadata[cluster_id].clone();
            let cluster_data_offset = self.cluster_data_offset;

            // Spawn parallel fetch for each cluster
            let future = tokio::spawn(async move {
                let path = object_store::path::Path::from(index_path);
                let range = (cluster_data_offset + meta.offset)
                    ..(cluster_data_offset + meta.offset + meta.size);

                let data = s3.get_range(&path, range).await?.to_vec();
                Ok::<_, Error>((cluster_id, data))
            });

            futures.push(future);
        }

        // Wait for all fetches in parallel
        let results = futures::future::join_all(futures).await;

        // Collect successful fetches
        let mut cluster_data = HashMap::new();
        for result in results {
            let (cluster_id, data) = result??;
            cluster_data.insert(cluster_id, data);
        }

        Ok(cluster_data)
    }

    /// Get codes for a specific cluster (called from C++ callback)
    fn get_codes_sync(&self, list_no: usize) -> &[u8] {
        // Access prefetched data synchronously
        // This works because prefetch already completed before C++ search
        let cache = self.prefetch_cache.blocking_read();
        cache.get(&list_no)
            .expect("Cluster should be prefetched")
            .as_slice()
    }
}
```

### Server Usage

```rust
// In your HTTP server (axum, actix-web, etc.)
async fn handle_search(
    State(index_registry): State<Arc<IndexRegistry>>,
    Json(req): Json<SearchRequest>,
) -> Result<Json<SearchResponse>, Error> {
    let index = index_registry.get(&req.index_name)?;

    // This is fully async - no blocking!
    let results = index.search(&req.query, req.nprobe, req.k).await?;

    Ok(Json(SearchResponse { results }))
}
```

### Pros

- **No `block_on`**: Entire search is async, works perfectly in tokio HTTP server
- **Maximum parallelism**: All clusters fetched concurrently
- **No deadlock risk**: Never blocks tokio worker threads
- **Clean API**: `async fn search()` is natural for HTTP server
- **Simple**: Prefetch completes before C++ callbacks run
- Uses `object_store` directly

### Cons

- Need to store prefetched data somewhere C++ can access
- Requires careful lifetime management of prefetched data
- C++ search must access prefetched data, not trigger new fetches

### Best For

- **Production HTTP server (strongly recommended)**
- Tokio-based async runtime
- Multiple concurrent searches
- Clean async/await code

## Comparison Matrix

Given constraints: MUST use `object_store` (async-only) + will run in tokio HTTP server

| Option | Complexity | Performance | Deadlock Risk in Server | Uses object_store | Parallelism | Best Use Case |
|--------|-----------|-------------|------------------------|-------------------|-------------|---------------|
| 1. Dedicated Runtime | Low | Poor | Medium | ✅ Yes | None | **Avoid** |
| 2. Shared Runtime | Medium | Medium | **FATAL** | ✅ Yes | None | **NEVER USE** |
| 3. Channel Bridge | High | Good | Medium | ✅ Yes | Sequential | On-demand only |
| 4. Thread Pool + Runtime | Medium-High | Good | Low | ✅ Yes | Per-fetch | Standalone CLI |
| 5. Async Search | **Low-Medium** | **Excellent** | **None** | ✅ Yes | **All clusters** | **Server (required)** |

**Critical difference for HTTP server**:
- Options 1-4 use `block_on`, which **will deadlock** when called from within a tokio runtime
- Option 5 makes search itself `async`, avoiding `block_on` entirely - **required for server context**

## Recommendation

### For HTTP Server Context (Production)

**MUST use Option 5: Async Search with Parallel Prefetch**

Reasons:
- **No deadlock**: Never uses `block_on` from within tokio runtime
- **Leverages Faiss architecture**: Uses the natural prefetch point between quantizer and search
- **Maximum parallelism**: Fetches all needed clusters concurrently with `join_all`
- **Clean API**: `async fn search()` is natural for HTTP handlers
- **Efficient**: One round-trip for all clusters instead of sequential fetches
- Uses `object_store` directly
- **Required for tokio HTTP server** - other options will deadlock

### Search Flow

```rust
// HTTP handler (axum/actix-web)
async fn search_handler(
    State(registry): State<Arc<IndexRegistry>>,
    Json(req): Json<SearchRequest>,
) -> Result<Json<SearchResponse>> {
    let index = registry.get(&req.index_name)?;

    // Fully async - no blocking
    let results = index.search(&req.query, req.nprobe, req.k).await?;

    Ok(Json(SearchResponse { results }))
}

// Inside index.search():
async fn search(&self, query: &[f32], nprobe: usize, k: usize) -> Result<SearchResults> {
    // Step 1: C++ quantizer (sync, fast)
    let cluster_ids = self.inner.quantizer_search(query, nprobe);

    // Step 2: Async prefetch (parallel, I/O-bound)
    let cluster_data = self.fetcher.prefetch_clusters_parallel(&cluster_ids).await?;

    // Step 3: Store for C++ callbacks
    self.cache.write().await.extend(cluster_data);

    // Step 4: C++ search (sync, uses cached data)
    let results = self.inner.search_preassigned(query, k, &cluster_ids);

    Ok(results)
}
```

### For Standalone CLI/Testing

**Can use Option 4: Thread Pool + Runtime** for simpler testing, but migrate to Option 5 for server.

## Implementation Plan

### Phase 2a: Async Search with Parallel Prefetch (No Cache)

1. **Modify Rust API** to expose async search:
   ```rust
   // lib.rs
   pub struct FaissIndexS3 {
       inner: Arc<FaissIVFIndexS3Wrapper>,
       fetcher: Arc<ClusterFetcher>,
       prefetch_cache: Arc<RwLock<HashMap<usize, Vec<u8>>>>,
   }

   impl FaissIndexS3 {
       pub async fn search(&self, query: &[f32], nprobe: usize, k: usize)
           -> Result<SearchResults>;
   }
   ```

2. **Create async `ClusterFetcher`**:
   - Store `Arc<dyn ObjectStore>` and cluster metadata
   - Implement `async fn prefetch_clusters_parallel()`:
     - Use `tokio::spawn` for each cluster
     - Use `futures::future::join_all` to wait for all
     - Return `HashMap<usize, Vec<u8>>`
   - **NO `block_on`** anywhere!

3. **Implement async search flow**:
   ```rust
   async fn search(&self, query: &[f32], nprobe: usize, k: usize) {
       // Step 1: C++ quantizer (sync)
       let cluster_ids = self.inner.quantizer_search(query, nprobe);

       // Step 2: Rust async prefetch
       let cluster_data = self.fetcher
           .prefetch_clusters_parallel(&cluster_ids).await?;

       // Step 3: Store in cache
       self.prefetch_cache.write().await.extend(cluster_data);

       // Step 4: C++ search (sync, reads from cache)
       self.inner.search_preassigned(query, k, &cluster_ids)
   }
   ```

4. **C++ callbacks access prefetched data**:
   ```cpp
   // S3RustIOInvertedLists::get_codes()
   // Calls Rust function that does blocking_read() on cache
   const uint8_t* get_codes(size_t list_no) {
       return rust_get_prefetched_codes(list_no);
   }
   ```

5. **Test with simple query in async context**:
   ```rust
   #[tokio::test]
   async fn test_async_search() {
       let index = create_index_from_s3().await?;
       let results = index.search(&query, 10, 5).await?;
       assert_eq!(results.len(), 5);
   }
   ```

### Phase 2b: Add Simple In-Memory Cache

1. Add `HashMap<usize, Vec<u8>>` to store previously fetched clusters
2. In `prefetch_clusters_parallel()`:
   - Check cache first
   - Only fetch missing clusters
   - Still parallel for cache misses
3. Add metrics: cache hits, misses, fetch count

### Phase 2c: Production (Optional)

1. Replace `HashMap` with LRU cache if memory is concern
2. Add proper error handling and retries
3. Add observability
4. Consider: Prefetch adjacent clusters for sequential access patterns

## Open Questions

1. **Prefetched data lifetime**: How to ensure prefetched data lives long enough for C++ callbacks?
   - Store in `Arc<RwLock<HashMap>>` per search?
   - Store per-index with generation/request ID?
2. **Error handling**: If one cluster fetch fails, fail entire search or return partial results?
3. **Concurrency limit**: Limit parallel fetches (e.g., max 10 concurrent with semaphore) or fetch all nprobe clusters?
4. **Cache in Phase 2b**: Simple `HashMap` or use `LRU` from the start?
5. **Request isolation**: How to prevent concurrent searches from interfering with each other's prefetched data?
   - Per-search cache?
   - Request ID tagging?
6. **Metrics**: What to track? (prefetch time, individual fetch latency, cluster sizes, cache effectiveness, etc.)

## References

- C++ implementation: `server/S3InvertedLists.cpp`
- Faiss search flow: `faiss/IndexIVF.cpp::search_preassigned()`
- object_store docs: https://docs.rs/object_store/
- reqwest blocking: https://docs.rs/reqwest/latest/reqwest/blocking/
- ureq: https://docs.rs/ureq/
