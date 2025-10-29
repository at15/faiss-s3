#pragma once

#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <aws/core/Aws.h>
#include <aws/core/utils/StringUtils.h>
#include <aws/s3-crt/S3CrtClient.h>
#include <aws/s3-crt/model/GetObjectRequest.h>

#include "constants.h"

namespace faiss_s3 {

/**
 * Creates an S3 client configured for the environment.
 *
 * Configuration is read from environment variables:
 * - S3_ENDPOINT_URL: Custom S3 endpoint (for S3Mock, MinIO, etc.)
 * - AWS_REGION: AWS region (defaults to us-east-1)
 *
 * @return Shared pointer to configured S3CrtClient
 */
inline std::shared_ptr<Aws::S3Crt::S3CrtClient> CreateS3Client() {
  Aws::S3Crt::ClientConfiguration config;

  // Check for custom endpoint (for S3Mock or MinIO)
  const char *endpoint = std::getenv(kEnvS3EndpointURL);
  if (endpoint) {
    config.endpointOverride = endpoint;
    std::cout << "[S3] Using custom endpoint: " << endpoint << std::endl;
  }

  // Check for region
  const char *region = std::getenv(kEnvAWSRegion);
  if (region) {
    config.region = region;
  } else {
    config.region = kDefaultAWSRegion;
  }

  return std::make_shared<Aws::S3Crt::S3CrtClient>(config);
}

/**
 * Downloads a byte range from S3 using HTTP range requests.
 *
 * This function performs a single GetObject request with a Range header
 * to fetch only the specified byte range from the S3 object.
 *
 * @param client S3 client (can be shared_ptr<void> for type erasure or typed)
 * @param bucket S3 bucket name
 * @param key S3 object key
 * @param offset Starting byte offset (0-indexed)
 * @param size Number of bytes to download
 * @return Vector containing the downloaded bytes
 * @throws std::runtime_error if S3 request fails or size mismatch occurs
 */
template <typename ClientPtr>
inline std::vector<uint8_t>
DownloadRangeFromS3(ClientPtr client_ptr, const std::string &bucket,
                    const std::string &key, size_t offset, size_t size) {
  // Handle both shared_ptr<void> (type-erased) and shared_ptr<S3CrtClient>
  std::shared_ptr<Aws::S3Crt::S3CrtClient> client;
  if constexpr (std::is_same_v<ClientPtr, std::shared_ptr<void>>) {
    client = std::static_pointer_cast<Aws::S3Crt::S3CrtClient>(client_ptr);
  } else {
    client = client_ptr;
  }

  Aws::S3Crt::Model::GetObjectRequest request;
  request.SetBucket(bucket);
  request.SetKey(key);

  // S3 range format: "bytes=start-end" (end is inclusive)
  // Using AWS string utilities for consistency with AWS SDK types
  Aws::String range = "bytes=" + Aws::Utils::StringUtils::to_string(offset) +
                      "-" +
                      Aws::Utils::StringUtils::to_string(offset + size - 1);
  request.SetRange(range);

  auto outcome = client->GetObject(request);

  if (!outcome.IsSuccess()) {
    std::ostringstream error_msg;
    error_msg << "S3 GetObject failed for s3://" << bucket << "/" << key
              << " [" << offset << ":" << (offset + size - 1)
              << "]: " << outcome.GetError().GetMessage();

    std::cerr << "[S3] " << error_msg.str() << std::endl;
    throw std::runtime_error(error_msg.str());
  }

  // Read response body
  auto &stream = outcome.GetResultWithOwnership().GetBody();
  std::vector<uint8_t> data(size);
  stream.read(reinterpret_cast<char *>(data.data()), size);

  // Verify we got the expected number of bytes
  size_t bytes_read = stream.gcount();
  if (bytes_read != size) {
    std::ostringstream error_msg;
    error_msg << "S3 read size mismatch: expected " << size
              << " bytes, got " << bytes_read;

    std::cerr << "[S3] Warning: " << error_msg.str() << std::endl;
    throw std::runtime_error(error_msg.str());
  }

  std::cout << "[S3] Downloaded range from s3://" << bucket << "/" << key
            << " [" << offset << ":" << (offset + size - 1) << "] "
            << "(" << (size / kBytesPerKB) << " KB)" << std::endl;

  return data;
}

} // namespace faiss_s3
