#include <iostream>

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
  // Return actual size because we are using it to calculate cluster data
  // offsets.
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

/**
 * IO Hook for reading ArrayInvertedLists.
 */
struct S3ReadNothingIOHook : faiss::InvertedListsIOHook {
  S3ReadNothingIOHook()
      : InvertedListsIOHook("ils3", typeid(S3ReadNothingInvertedLists).name()) {
    std::cout << "[S3ReadNothingIOHook] Registering S3ReadNothingIOHook "
                 "(fourcc: ils3)"
              << std::endl;
  }

  void write(const faiss::InvertedLists *ils,
             faiss::IOWriter *f) const override {
    FAISS_THROW_MSG("S3ReadNothingInvertedLists is read-only, cannot write");
  }

  faiss::InvertedLists *read(faiss::IOReader *f, int io_flags) const override {
    FAISS_THROW_MSG("Use read_ArrayInvertedLists instead");
  }

  faiss::InvertedLists *
  read_ArrayInvertedLists(faiss::IOReader *f, int io_flags, size_t nlist,
                          size_t code_size,
                          const std::vector<size_t> &sizes) const override {
    std::cout << "[S3ReadNothingIOHook] nlist=" << nlist
              << ", code_size=" << code_size << std::endl;

    // Save the sizes for each cluster
    auto s3il = new S3ReadNothingInvertedLists(nlist, code_size, sizes);
    return s3il;
  }
};

void RegisterS3ReadNothingIOHook() {
  static bool registered = false;
  if (!registered) {
    faiss::InvertedListsIOHook::add_callback(new S3ReadNothingIOHook());
    std::cout << "[S3ReadNothingIOHook] S3ReadNothingIOHook registered"
              << std::endl;
    registered = true;
  }
}
} // namespace faiss_s3