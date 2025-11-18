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
struct S3ReadNothingInvertedLists : faiss::ReadOnlyInvertedLists {
  std::vector<size_t> cluster_sizes; // Store sizes from hook

  S3ReadNothingInvertedLists(size_t nlist, size_t code_size,
                             const std::vector<size_t> &sizes);

  size_t list_size(size_t list_no) const override;
  const uint8_t *get_codes(size_t list_no) const override;
  const idx_t *get_ids(size_t list_no) const override;
};

// TODO: Actual implementation that accepts callback for S3 fetching logic

// TODO: check the prefetch logic, seems to be useful to allow us not calling
// search_preassigned directly
// https://github.com/facebookresearch/faiss/blob/221b5c2450f80d0710bbe392be8c48ead6bd1a9d/faiss/IndexIVF.cpp#L300-L337
// /// prepare the following lists (default does nothing)
// /// a list can be -1 hence the signed long
// virtual void prefetch_lists(const idx_t* list_nos, int nlist) const;
} // namespace faiss_s3
