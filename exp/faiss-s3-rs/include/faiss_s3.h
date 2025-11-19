#pragma once

#include "rust/cxx.h" // generated in target/cxxbridge/rust/cxx.h

/**
 * Create an example IVF index and write it to the given file.
 */
void CreateExampleIVFIndex(rust::Str index_file_name);

/**
 * Search and example IVF index
 */
void SearchExampleIVFIndex(rust::Str index_file_name);

/**
 * Calculate the offset of inverted list cluster data in a IVF index file.
 * This allow loading the index without the cluster data part later.
 */
size_t GetClusterDataOffset(rust::Str index_file_name);