#pragma once

#include <cstddef>
#include <cstdint>

namespace faiss_s3 {

// Network Configuration
constexpr int DEFAULT_SERVER_PORT = 9001;
constexpr int LISTEN_BACKLOG = 10;

// Protocol Limits
constexpr size_t MAX_COMMAND_LINE_LENGTH = 8192;    // Maximum command line size
constexpr uint32_t MAX_BINARY_ARRAY_SIZE = 100 * 1024 * 1024; // 100MB max per array

// Cache Configuration
constexpr size_t DEFAULT_CACHE_SIZE_MB = 2048;      // Default: 2GB
constexpr size_t UNLIMITED_CACHE = 0;               // 0 = unlimited cache

// Memory Conversions
constexpr size_t BYTES_PER_KB = 1024;
constexpr size_t BYTES_PER_MB = 1024 * 1024;
constexpr size_t KB_PER_MB = 1024;

// AWS Configuration
constexpr const char* DEFAULT_AWS_REGION = "us-east-1";
constexpr const char* ENV_S3_ENDPOINT_URL = "S3_ENDPOINT_URL";
constexpr const char* ENV_AWS_REGION = "AWS_REGION";
constexpr const char* ENV_CACHE_SIZE_MB = "FAISS_S3_CACHE_SIZE_MB";

} // namespace faiss_s3
