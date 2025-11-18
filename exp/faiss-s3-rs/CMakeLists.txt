cmake_minimum_required(VERSION 3.17 FATAL_ERROR)
project (FAISS-S3-CPP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# TODO: Change to release, get it automatically from cargo's env/flag?
set(CMAKE_BUILD_TYPE Debug)


set(CARGO_MANIFEST ${CMAKE_SOURCE_DIR}/Cargo.toml)
set(CARGO_TARGET_DIR ${CMAKE_SOURCE_DIR}/target)

# Add faiss we built from source
set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};${CMAKE_SOURCE_DIR}/deps/faiss-home")

# Deps
find_package(OpenMP REQUIRED)
find_package(faiss REQUIRED)

# NOTE: We need to add the cxx bridge into the library
set(BRIDGE_CPP ${CARGO_TARGET_DIR}/cxxbridge/faiss-s3-rs/src/lib.rs.cc)

add_library(faiss-s3-cpp
    src/faiss_s3.cpp
    src/s3_inverted_lists.cpp
    ${BRIDGE_CPP}
)

target_include_directories(
    faiss-s3-cpp
    PRIVATE
    include/
    ${CARGO_TARGET_DIR}/cxxbridge/faiss-s3-rs/src/ # lib.rs.h lib.rs.cc
    ${CARGO_TARGET_DIR}/cxxbridge/ # rust/cxx.h
)

# TODO: Add this to find faiss header files
target_link_libraries(
   faiss-s3-cpp
   faiss
)