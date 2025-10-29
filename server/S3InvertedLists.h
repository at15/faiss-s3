#pragma once

#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <faiss/index_io.h>
#include <faiss/invlists/InvertedLists.h>

namespace faiss_s3 {

// Type alias for Faiss index type to avoid polluting namespace
using idx_t = faiss::idx_t;

// Placeholder created during read_index() with IO_FLAG_S3
// All methods throw errors until replaced with S3ReadOnlyInvertedLists
struct S3ReadNothingInvertedLists : faiss::InvertedLists {
  std::vector<size_t> cluster_sizes; // Store sizes from hook

  S3ReadNothingInvertedLists(size_t nlist, size_t code_size,
                             const std::vector<size_t> &sizes);

  size_t list_size(size_t list_no) const override;
  const uint8_t *get_codes(size_t list_no) const override;
  const idx_t *get_ids(size_t list_no) const override;

  size_t add_entries(size_t list_no, size_t n_entry, const idx_t *ids_in,
                     const uint8_t *code) override;

  void update_entries(size_t list_no, size_t offset, size_t n_entry,
                      const idx_t *ids_in, const uint8_t *code) override;

  void resize(size_t list_no, size_t new_size) override;
};

// S3OnDemandInvertedLists - loads cluster data from S3 on-demand with caching
// Simplified version that merges S3 client and caching logic into one class
struct S3OnDemandInvertedLists : faiss::InvertedLists {
  // Constructor
  S3OnDemandInvertedLists(std::shared_ptr<void> s3_client,
                          const std::string &bucket, const std::string &key,
                          size_t cluster_data_offset, size_t nlist,
                          size_t code_size,
                          const std::vector<size_t> &cluster_sizes);

  // Read methods
  size_t list_size(size_t list_no) const override;
  const uint8_t *get_codes(size_t list_no) const override;
  const idx_t *get_ids(size_t list_no) const override;

  // Write methods (not supported, read-only)
  size_t add_entries(size_t list_no, size_t n_entry, const idx_t *ids_in,
                     const uint8_t *code) override;

  void update_entries(size_t list_no, size_t offset, size_t n_entry,
                      const idx_t *ids_in, const uint8_t *code) override;

  void resize(size_t list_no, size_t new_size) override;

  // Cache statistics
  size_t cache_size() const;
  void clear_cache();
  size_t cache_hits() const;
  size_t cache_misses() const;
  size_t cache_bytes() const;

  // Cache configuration
  void set_max_cache_bytes(size_t max_bytes);
  size_t get_max_cache_bytes() const;

private:
  // S3 configuration
  std::shared_ptr<void> s3_client_;
  std::string s3_bucket_;
  std::string s3_key_;
  size_t cluster_data_offset_;

  // Cluster metadata
  std::vector<size_t> cluster_sizes_;

  // Cache for cluster data
  struct ClusterData {
    std::vector<uint8_t> codes;
    std::vector<idx_t> ids;
  };
  mutable std::unordered_map<size_t, std::shared_ptr<ClusterData>> cache_;
  mutable std::mutex cache_mutex_;

  // Cache statistics
  mutable size_t cache_hits_ = 0;
  mutable size_t cache_misses_ = 0;
  mutable size_t cache_bytes_ = 0;

  // Cache limits and LRU tracking
  size_t max_cache_bytes_ = 0; // 0 = unlimited
  mutable std::list<size_t> lru_list_; // Front = most recently used
  mutable std::unordered_map<size_t, std::list<size_t>::iterator> lru_map_;

  // Helper methods
  std::shared_ptr<ClusterData> fetch_cluster(size_t list_no) const;
  size_t calculate_cluster_offset(size_t list_no) const;
  void evict_lru_if_needed(size_t bytes_needed) const;
};

// IO flag for S3 lazy loading
// 0x7333 = "s3" in hex (big-endian for fourcc)
// Combined with IO_FLAG_SKIP_IVF_DATA and "il" prefix → "ils3"
// IO flag for S3 on-demand loading
// fourcc("ils3") = 0x33736c69, so upper 16 bits = 0x3373
const int IO_FLAG_S3 = faiss::IO_FLAG_SKIP_IVF_DATA | 0x33730000;

// Manually register S3 hook (call this before using IO_FLAG_S3)
void register_s3_io_hook();

// TODO: Check IndexIVFlatPanorama
// https://github.com/facebookresearch/faiss/pull/4606

} // namespace faiss_s3