# Copied from rocksdb_ivf

cmake_minimum_required(VERSION 3.17 FATAL_ERROR)
project (FAISS_S3)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Debug)

# Deps
find_package(OpenMP REQUIRED)
find_package(faiss REQUIRED)
find_package(AWSSDK REQUIRED COMPONENTS s3-crt)

add_executable(s3_cache_server s3_cache_server.cpp S3InvertedLists.cpp)

# TODO: ld: warning: ignoring duplicate libraries: '-ldl', '-lpthread'
target_link_libraries(s3_cache_server faiss ${AWSSDK_LINK_LIBRARIES} pthread)
