#include <cassert>
#include <cstdlib>
#include <iostream>
#include <sstream>

#include <faiss/impl/FaissAssert.h>
#include <faiss/invlists/InvertedListsIOHook.h>

#include "S3InvertedLists.h"
#include "constants.h"
#include "s3_utils.h"

namespace faiss_s3 {

// Type alias for Faiss index type to avoid polluting namespace
using idx_t = faiss::idx_t;

// ============================================================================
// S3ReadNothingInvertedLists - Placeholder
// ============================================================================

S3ReadNothingInvertedLists::S3ReadNothingInvertedLists(
    size_t nlist, size_t code_size, const std::vector<size_t> &sizes)
    : faiss::InvertedLists(nlist, code_size), cluster_sizes(sizes) {}

size_t S3ReadNothingInvertedLists::list_size(size_t list_no) const {
  // Return actual size so index stats work before replacement
  return cluster_sizes[list_no];
}

const uint8_t *S3ReadNothingInvertedLists::get_codes(size_t list_no) const {
  FAISS_THROW_MSG(
      "S3ReadNothingInvertedLists: not initialized, use replace_invlists");
}

const idx_t *S3ReadNothingInvertedLists::get_ids(size_t list_no) const {
  FAISS_THROW_MSG(
      "S3ReadNothingInvertedLists: not initialized, use replace_invlists");
}

size_t S3ReadNothingInvertedLists::add_entries(size_t, size_t, const idx_t *,
                                               const uint8_t *) {
  FAISS_THROW_MSG("S3ReadNothingInvertedLists: read-only");
}

void S3ReadNothingInvertedLists::update_entries(size_t, size_t, size_t,
                                                const idx_t *,
                                                const uint8_t *) {
  FAISS_THROW_MSG("S3ReadNothingInvertedLists: read-only");
}

void S3ReadNothingInvertedLists::resize(size_t, size_t) {
  FAISS_THROW_MSG("S3ReadNothingInvertedLists: read-only");
}

// ============================================================================
// S3OnDemandInvertedLists - Simplified Implementation
// ============================================================================

S3OnDemandInvertedLists::S3OnDemandInvertedLists(
    std::shared_ptr<void> s3_client, const std::string &bucket,
    const std::string &key, size_t cluster_data_offset, size_t nlist,
    size_t code_size, const std::vector<size_t> &cluster_sizes)
    : faiss::InvertedLists(nlist, code_size), s3_client_(s3_client),
      s3_bucket_(bucket), s3_key_(key),
      cluster_data_offset_(cluster_data_offset), cluster_sizes_(cluster_sizes) {
}

size_t S3OnDemandInvertedLists::list_size(size_t list_no) const {
  return cluster_sizes_[list_no];
}

const uint8_t *S3OnDemandInvertedLists::get_codes(size_t list_no) const {
  auto data = fetch_cluster(list_no);
  return data->codes.data();
}

const idx_t *S3OnDemandInvertedLists::get_ids(size_t list_no) const {
  auto data = fetch_cluster(list_no);
  return data->ids.data();
}

size_t S3OnDemandInvertedLists::add_entries(size_t list_no, size_t n_entry,
                                            const idx_t *ids_in,
                                            const uint8_t *code) {
  FAISS_THROW_MSG("S3OnDemandInvertedLists: read-only");
}

void S3OnDemandInvertedLists::update_entries(size_t list_no, size_t offset,
                                             size_t n_entry,
                                             const idx_t *ids_in,
                                             const uint8_t *code) {
  FAISS_THROW_MSG("S3OnDemandInvertedLists: read-only");
}

void S3OnDemandInvertedLists::resize(size_t list_no, size_t new_size) {
  FAISS_THROW_MSG("S3OnDemandInvertedLists: read-only");
}

size_t S3OnDemandInvertedLists::cache_size() const {
  std::lock_guard<std::mutex> lock(cache_mutex_);
  return cache_.size();
}

void S3OnDemandInvertedLists::clear_cache() {
  std::lock_guard<std::mutex> lock(cache_mutex_);
  cache_.clear();
  lru_list_.clear();
  lru_map_.clear();
  cache_hits_ = 0;
  cache_misses_ = 0;
  cache_bytes_ = 0;
}

size_t S3OnDemandInvertedLists::cache_hits() const {
  std::lock_guard<std::mutex> lock(cache_mutex_);
  return cache_hits_;
}

size_t S3OnDemandInvertedLists::cache_misses() const {
  std::lock_guard<std::mutex> lock(cache_mutex_);
  return cache_misses_;
}

size_t S3OnDemandInvertedLists::cache_bytes() const {
  std::lock_guard<std::mutex> lock(cache_mutex_);
  return cache_bytes_;
}

void S3OnDemandInvertedLists::set_max_cache_bytes(size_t max_bytes) {
  std::lock_guard<std::mutex> lock(cache_mutex_);
  max_cache_bytes_ = max_bytes;
  std::cout << "[Cache] Set max cache size to " << (max_bytes / BYTES_PER_MB)
            << " MB" << std::endl;

  // Trigger eviction if we're over the limit
  if (max_bytes > UNLIMITED_CACHE) {
    evict_lru_if_needed(0);
  }
}

size_t S3OnDemandInvertedLists::get_max_cache_bytes() const {
  return max_cache_bytes_;
}

void S3OnDemandInvertedLists::evict_lru_if_needed(size_t bytes_needed) const {
  // If no limit, nothing to evict
  if (max_cache_bytes_ == UNLIMITED_CACHE) {
    return;
  }

  // Evict least recently used items until we have enough space
  while (!lru_list_.empty() && cache_bytes_ + bytes_needed > max_cache_bytes_) {
    // Get least recently used cluster (back of list)
    size_t evict_list_no = lru_list_.back();
    lru_list_.pop_back();
    lru_map_.erase(evict_list_no);

    // Remove from cache and update byte count
    auto cache_it = cache_.find(evict_list_no);
    if (cache_it != cache_.end()) {
      size_t evicted_bytes = cache_it->second->codes.size() +
                             cache_it->second->ids.size() * sizeof(idx_t);
      cache_bytes_ -= evicted_bytes;
      cache_.erase(cache_it);

      std::cout << "[Cache] Evicted cluster " << evict_list_no << " ("
                << (evicted_bytes / BYTES_PER_KB) << " KB), cache now "
                << (cache_bytes_ / BYTES_PER_MB) << " MB" << std::endl;
    }
  }
}

size_t S3OnDemandInvertedLists::calculate_cluster_offset(size_t list_no) const {
  size_t offset = cluster_data_offset_;

  // Sum up all previous clusters' sizes
  for (size_t i = 0; i < list_no; i++) {
    size_t n = cluster_sizes_[i];
    if (n > 0) {
      offset += n * code_size;     // codes
      offset += n * sizeof(idx_t); // ids
    }
  }

  return offset;
}

std::shared_ptr<S3OnDemandInvertedLists::ClusterData>
S3OnDemandInvertedLists::fetch_cluster(size_t list_no) const {
  // First check: with lock held
  {
    std::lock_guard<std::mutex> lock(cache_mutex_);

    // Check if already cached
    auto it = cache_.find(list_no);
    if (it != cache_.end()) {
      cache_hits_++;

      // Move to front of LRU list (most recently used)
      auto lru_it = lru_map_.find(list_no);
      if (lru_it != lru_map_.end()) {
        lru_list_.erase(lru_it->second);
        lru_list_.push_front(list_no);
        lru_map_[list_no] = lru_list_.begin();
      }

      return it->second;
    }

    // Record cache miss
    cache_misses_++;
  }

  // Cache miss - prepare to fetch from S3 (without holding lock)
  size_t n = cluster_sizes_[list_no];
  auto data = std::make_shared<ClusterData>();

  if (n == 0) {
    // Empty cluster - cache it and return
    std::lock_guard<std::mutex> lock(cache_mutex_);
    // Double-check in case another thread cached it
    auto it = cache_.find(list_no);
    if (it != cache_.end()) {
      return it->second;
    }
    cache_[list_no] = data;
    lru_list_.push_front(list_no);
    lru_map_[list_no] = lru_list_.begin();
    return data;
  }

  size_t offset = calculate_cluster_offset(list_no);
  size_t codes_bytes = n * code_size;
  size_t ids_bytes = n * sizeof(idx_t);
  size_t total_bytes = codes_bytes + ids_bytes;

  std::cout << "→ Fetching cluster " << list_no << " (" << n << " vectors, "
            << total_bytes << " bytes)" << std::endl;

  // Download cluster data from S3 WITHOUT holding the mutex
  // This allows other threads to access the cache while we're downloading
  auto raw_data =
      DownloadRangeFromS3(s3_client_, s3_bucket_, s3_key_, offset, total_bytes);

  // Split into codes and ids
  data->codes.resize(codes_bytes);
  data->ids.resize(n);

  memcpy(data->codes.data(), raw_data.data(), codes_bytes);
  memcpy(data->ids.data(), raw_data.data() + codes_bytes, ids_bytes);

  // Now acquire lock to update cache
  {
    std::lock_guard<std::mutex> lock(cache_mutex_);

    // Double-check: another thread might have fetched this while we were
    // downloading
    auto it = cache_.find(list_no);
    if (it != cache_.end()) {
      std::cout << "✓ Cluster " << list_no
                << " was cached by another thread, using that" << std::endl;
      return it->second;
    }

    // Evict LRU items if needed to make space
    evict_lru_if_needed(total_bytes);

    std::cout << "✓ Cached cluster " << list_no
              << " (cache: " << (cache_bytes_ / BYTES_PER_MB) << " MB)"
              << std::endl;

    // Cache the data and track in LRU
    cache_[list_no] = data;
    cache_bytes_ += total_bytes;
    lru_list_.push_front(list_no);
    lru_map_[list_no] = lru_list_.begin();

    return data;
  }
}

// ============================================================================
// S3InvertedListsIOHook - IO Hook Registration
// ============================================================================

struct S3InvertedListsIOHook : faiss::InvertedListsIOHook {
  S3InvertedListsIOHook()
      : InvertedListsIOHook("ils3", typeid(S3ReadNothingInvertedLists).name()) {
    std::cout << "[S3Hook] Registering S3InvertedListsIOHook (fourcc: ils3)"
              << std::endl;
  }

  void write(const faiss::InvertedLists *ils,
             faiss::IOWriter *f) const override {
    FAISS_THROW_MSG("S3InvertedLists is read-only, cannot write");
  }

  faiss::InvertedLists *read(faiss::IOReader *f, int io_flags) const override {
    FAISS_THROW_MSG("Use read_ArrayInvertedLists instead");
  }

  faiss::InvertedLists *
  read_ArrayInvertedLists(faiss::IOReader *f, int io_flags, size_t nlist,
                          size_t code_size,
                          const std::vector<size_t> &sizes) const override {
    std::cout << "[S3Hook] Creating S3ReadNothingInvertedLists placeholder"
              << std::endl;
    std::cout << "[S3Hook] nlist=" << nlist << ", code_size=" << code_size
              << std::endl;

    // Create placeholder with sizes
    auto s3il = new S3ReadNothingInvertedLists(nlist, code_size, sizes);

    // Calculate total data size to skip
    size_t total_bytes = 0;
    for (size_t s : sizes) {
      total_bytes += s * (code_size + sizeof(faiss::idx_t));
    }

    std::cout << "[S3Hook] Skipping " << total_bytes << " bytes of cluster data"
              << std::endl;

    // Skip cluster data in file stream
    auto *reader = dynamic_cast<faiss::FileIOReader *>(f);
    if (reader) {
      fseek(reader->f, total_bytes, SEEK_CUR);
    }

    return s3il;
  }
};

// Public function to manually register hook
void register_s3_io_hook() {
  static bool registered = false;
  if (!registered) {
    faiss::InvertedListsIOHook::add_callback(new S3InvertedListsIOHook());
    std::cout << "[S3Hook] S3InvertedListsIOHook registered (manual)"
              << std::endl;
    registered = true;
  }
}

} // namespace faiss_s3