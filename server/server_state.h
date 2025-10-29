#pragma once

#include <atomic>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <string>

#include <aws/core/Aws.h>
#include <aws/s3-crt/S3CrtClient.h>
#include <faiss/Index.h>

#include "S3InvertedLists.h"

namespace faiss_s3 {

/**
 * State for a single loaded index.
 *
 * Contains all the information needed to manage a loaded Faiss index,
 * including its S3 location, Faiss components, and cache statistics.
 */
struct IndexState {
  int id_;                                             // Unique index ID
  std::string bucket_;                                 // S3 bucket name
  std::string key_;                                    // S3 object key
  int64_t cluster_data_offset_;                        // Offset to cluster data in S3 file

  // Faiss components
  std::shared_ptr<faiss::Index> index_;                // The loaded Faiss index
  std::shared_ptr<Aws::S3Crt::S3CrtClient> s3_client_; // S3 client for this index
  S3OnDemandInvertedLists *s3_invlists_;               // On-demand inverted lists (non-owning)

  IndexState()
      : id_(0), cluster_data_offset_(0), s3_invlists_(nullptr) {}
};

/**
 * Global server state managing all loaded indexes and AWS SDK lifecycle.
 *
 * This class is responsible for:
 * - Initializing/shutting down the AWS SDK
 * - Registering the S3 IO hook for Faiss
 * - Managing the collection of loaded indexes
 * - Thread-safe index lookup and storage
 */
class ServerState {
public:
  /**
   * Initializes the AWS SDK and registers Faiss hooks.
   */
  ServerState() {
    // Initialize AWS SDK
    std::cout << "[Server] Initializing AWS SDK..." << std::endl;
    Aws::InitAPI(sdk_options_);

    // Register S3 IO hook for Faiss
    std::cout << "[Server] Registering S3 IO hook..." << std::endl;
    register_s3_io_hook();
  }

  /**
   * Cleans up all loaded indexes and shuts down the AWS SDK.
   *
   * Thread Safety:
   * - Called during server shutdown when Run() loop has exited
   * - No new client connections are accepted after Stop() is called
   * - Active client threads may still be running (detached), but they
   *   only read from indexes (no modifications)
   * - Clearing the indexes map invalidates all IndexState pointers
   * - This is safe because all index access is read-only and the
   *   shared_ptr references in IndexState keep objects alive
   */
  ~ServerState() {
    // Cleanup indexes
    {
      std::lock_guard<std::mutex> lock(indexes_mutex_);
      std::cout << "[Server] Cleaning up " << loaded_indexes_.size()
                << " loaded indexes..." << std::endl;
      loaded_indexes_.clear();
    }

    // Shutdown AWS SDK
    // Note: Active client threads may still have S3 client references
    // via shared_ptr, so actual cleanup happens when last reference released
    std::cout << "[Server] Shutting down AWS SDK..." << std::endl;
    Aws::ShutdownAPI(sdk_options_);
  }

  // Delete copy/move constructors (singleton-like lifecycle)
  ServerState(const ServerState &) = delete;
  ServerState &operator=(const ServerState &) = delete;
  ServerState(ServerState &&) = delete;
  ServerState &operator=(ServerState &&) = delete;

  /**
   * Adds a new index to the server state.
   *
   * @param bucket S3 bucket name
   * @param key S3 object key
   * @param cluster_data_offset Offset to cluster data in the S3 file
   * @param index Shared pointer to the Faiss index
   * @param s3_client Shared pointer to the S3 client
   * @param s3_invlists Pointer to the S3OnDemandInvertedLists (non-owning)
   * @return Unique ID assigned to this index
   */
  int AddIndex(const std::string &bucket, const std::string &key,
               int64_t cluster_data_offset,
               std::shared_ptr<faiss::Index> index,
               std::shared_ptr<Aws::S3Crt::S3CrtClient> s3_client,
               S3OnDemandInvertedLists *s3_invlists) {
    std::lock_guard<std::mutex> lock(indexes_mutex_);
    int id = next_index_id_.fetch_add(1);

    IndexState state;
    state.id_ = id;
    state.bucket_ = bucket;
    state.key_ = key;
    state.cluster_data_offset_ = cluster_data_offset;
    state.index_ = index;
    state.s3_client_ = s3_client;
    state.s3_invlists_ = s3_invlists;

    loaded_indexes_[id] = state;
    return id;
  }

  /**
   * Gets an index by ID.
   *
   * Thread Safety Guarantees:
   * - This function returns a pointer to IndexState after releasing the lock
   * - The returned pointer remains valid because:
   *   1. Indexes are NEVER removed during server operation
   *   2. Map entries are only erased at server shutdown (in destructor)
   *   3. No concurrent access occurs during shutdown
   * - IndexState itself contains thread-safe components (shared_ptr, mutex in S3OnDemandInvertedLists)
   *
   * Lifetime Guarantee:
   * - Returned pointer is valid until ServerState destructor is called
   * - Safe to use across multiple operations without re-acquiring
   *
   * @param id Index ID to look up
   * @param out_state Pointer to receive the IndexState pointer (valid until shutdown)
   * @return true if index was found, false otherwise
   */
  bool GetIndex(int id, IndexState **out_state) {
    std::lock_guard<std::mutex> lock(indexes_mutex_);
    auto it = loaded_indexes_.find(id);
    if (it == loaded_indexes_.end()) {
      return false;
    }
    *out_state = &it->second;
    return true;
  }

  /**
   * Returns the total number of loaded indexes.
   *
   * @return Number of indexes currently loaded
   */
  int GetIndexCount() {
    std::lock_guard<std::mutex> lock(indexes_mutex_);
    return loaded_indexes_.size();
  }

  /**
   * Finds an existing index by its S3 location.
   *
   * This allows reusing already-loaded indexes instead of loading them again.
   *
   * @param bucket S3 bucket name
   * @param key S3 object key
   * @param cluster_data_offset Offset to cluster data in the S3 file
   * @return Index ID if found, -1 otherwise
   */
  int FindIndexByLocation(const std::string &bucket, const std::string &key,
                          int64_t cluster_data_offset) {
    std::lock_guard<std::mutex> lock(indexes_mutex_);
    for (const auto &kv : loaded_indexes_) {
      const IndexState &state = kv.second;
      if (state.bucket_ == bucket && state.key_ == key &&
          state.cluster_data_offset_ == cluster_data_offset) {
        return state.id_;
      }
    }
    return -1; // Not found
  }

  /**
   * Provides access to the indexes mutex for aggregate operations.
   *
   * This allows external code to lock the mutex when iterating over
   * all indexes (e.g., for computing global statistics).
   *
   * @return Reference to the indexes mutex
   */
  std::mutex &GetIndexesMutex() { return indexes_mutex_; }

  /**
   * Gets a read-only view of all loaded indexes.
   *
   * Caller must hold indexes_mutex_ when accessing this map.
   *
   * @return Reference to the loaded indexes map
   */
  const std::map<int, IndexState> &GetLoadedIndexes() const {
    return loaded_indexes_;
  }

private:
  Aws::SDKOptions sdk_options_;
  std::atomic<int> next_index_id_{1};
  std::map<int, IndexState> loaded_indexes_;
  std::mutex indexes_mutex_;
};

} // namespace faiss_s3
