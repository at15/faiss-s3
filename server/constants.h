#pragma once

#include <cstddef>
#include <cstdint>

namespace faiss_s3 {

// Network Configuration
constexpr int kDefaultServerPort = 9001;
constexpr int kListenBacklog = 10;

// Protocol Limits
constexpr size_t kMaxCommandLineLength = 8192;    // Maximum command line size
constexpr uint32_t kMaxBinaryArraySize = 100 * 1024 * 1024; // 100MB max per array

// Cache Configuration
constexpr size_t kDefaultCacheSizeMB = 2048;      // Default: 2GB
constexpr size_t kUnlimitedCache = 0;             // 0 = unlimited cache

// Memory Conversions
constexpr size_t kBytesPerKB = 1024;
constexpr size_t kBytesPerMB = 1024 * 1024;
constexpr size_t kKBPerMB = 1024;

// AWS Configuration
constexpr const char* kDefaultAWSRegion = "us-east-1";
constexpr const char* kEnvS3EndpointURL = "S3_ENDPOINT_URL";
constexpr const char* kEnvAWSRegion = "AWS_REGION";
constexpr const char* kEnvCacheSizeMB = "FAISS_S3_CACHE_SIZE_MB";

} // namespace faiss_s3
