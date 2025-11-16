#pragma once

#include "rust/cxx.h" // generated in target/cxxbridge/rust/cxx.h

/**
 * Create an example IVF index and write it to the given file.
 */
void create_example_ivf_index(rust::Str index_file_name);