#include <faiss/impl/FaissAssert.h>
#include <faiss/invlists/InvertedListsIOHook.h>

#include "s3_inverted_lists.h"

namespace faiss_s3 {

// Type alias for Faiss index type to avoid polluting namespace
using idx_t = faiss::idx_t;

S3ReadNothingInvertedLists::S3ReadNothingInvertedLists(
    size_t nlist, size_t code_size, const std::vector<size_t> &sizes)
    : faiss::ReadOnlyInvertedLists(nlist, code_size), cluster_sizes(sizes) {}

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
} // namespace faiss_s3