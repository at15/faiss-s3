#pragma once

#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <unistd.h>
#include <vector>

#include <faiss/IndexIVFFlat.h>
#include <faiss/impl/io.h>
#include <faiss/index_io.h>

#include "S3InvertedLists.h"
#include "binary_io.h"
#include "constants.h"
#include "protocol.h"
#include "s3_utils.h"
#include "server_state.h"

namespace faiss_s3 {

/**
 * Handles client connections and processes commands.
 *
 * Each client connection is handled by a separate instance of this class,
 * typically running in its own thread. The handler processes the text-based
 * protocol and delegates to specialized command handlers.
 *
 * Supported commands:
 * - ECHO: Echo a message back (for testing)
 * - LOAD: Load a Faiss index from S3
 * - SEARCH: Perform k-NN search on a loaded index
 * - INFO: Get statistics about cache or index
 */
class ClientHandler {
public:
  /**
   * Creates a client handler for a socket connection.
   *
   * @param socket_fd Client socket file descriptor (will be closed on destruction)
   * @param server_state Pointer to the global server state
   */
  ClientHandler(int socket_fd, ServerState *server_state)
      : socket_fd_(socket_fd), server_state_(server_state) {}

  /**
   * Destructor closes the client socket.
   */
  ~ClientHandler() {
    if (socket_fd_ >= 0) {
      close(socket_fd_);
    }
  }

  // Delete copy/move constructors (socket is non-copyable)
  ClientHandler(const ClientHandler &) = delete;
  ClientHandler &operator=(const ClientHandler &) = delete;
  ClientHandler(ClientHandler &&) = delete;
  ClientHandler &operator=(ClientHandler &&) = delete;

  /**
   * Main loop: reads and processes commands until disconnection.
   *
   * This function blocks until the client disconnects or an error occurs.
   */
  void Run() {
    std::cout << "[Client " << socket_fd_ << "] Connected" << std::endl;

    while (true) {
      // Read command line
      std::string line;
      if (!BinaryIO::ReadLine(socket_fd_, line)) {
        std::cout << "[Client " << socket_fd_ << "] Disconnected" << std::endl;
        break;
      }

      if (line.empty()) {
        continue; // Skip empty lines
      }

      // Parse and handle command
      auto params = ProtocolParser::ParseCommand(line);
      if (params.find("__command") == params.end()) {
        SendError("INVALID_COMMAND", "Empty command");
        continue;
      }

      std::string command = params["__command"];
      std::cout << "[Client " << socket_fd_ << "] Command: " << command
                << std::endl;

      // Dispatch to command handlers
      if (command == "ECHO") {
        HandleEcho(params);
      } else if (command == "LOAD") {
        HandleLoad(params);
      } else if (command == "SEARCH") {
        HandleSearch(params);
      } else if (command == "INFO") {
        HandleInfo(params);
      } else {
        SendError("INVALID_COMMAND", "Unknown command: " + command);
      }
    }
  }

private:
  int socket_fd_;
  ServerState *server_state_;

  /**
   * Sends a response string to the client.
   *
   * @param response Response string (should include trailing newline)
   * @return true if successful, false on error
   */
  bool SendResponse(const std::string &response) {
    return BinaryIO::WriteExact(socket_fd_, response.c_str(), response.size());
  }

  /**
   * Sends an error response to the client.
   *
   * @param code Error code (e.g., "INVALID_PARAM")
   * @param msg Human-readable error message
   * @return true if successful, false on error
   */
  bool SendError(const std::string &code, const std::string &msg) {
    std::string error = ProtocolParser::FormatError(code, msg);
    std::cout << "[Client " << socket_fd_ << "] Error: " << code << " - " << msg
              << std::endl;
    return SendResponse(error);
  }

  /**
   * Handles the ECHO command.
   *
   * Command: ECHO msg=<message>
   * Response: msg=<message>
   */
  void HandleEcho(const std::map<std::string, std::string> &params) {
    auto it = params.find("msg");
    if (it == params.end()) {
      SendError("MISSING_PARAM", "Missing msg parameter");
      return;
    }

    std::map<std::string, std::string> response;
    response["msg"] = it->second;
    SendResponse(ProtocolParser::FormatResponse(response));
  }

  /**
   * Handles the LOAD command.
   *
   * Loads a Faiss index from S3. The index metadata is downloaded and parsed,
   * but cluster data is loaded on-demand during searches.
   *
   * Command: LOAD bucket=<bucket> key=<key> cluster_data_offset=<offset>
   * Response: index=<id> d=<dimension> ntotal=<count> nlist=<clusters> metric_type=<type>
   *
   * If the index is already loaded, returns the existing index ID.
   */
  void HandleLoad(const std::map<std::string, std::string> &params) {
    // Parse parameters
    auto bucket_it = params.find("bucket");
    auto key_it = params.find("key");
    auto offset_it = params.find("cluster_data_offset");

    if (bucket_it == params.end() || key_it == params.end() ||
        offset_it == params.end()) {
      SendError("MISSING_PARAM",
                "Missing required parameters (bucket, key, cluster_data_offset)");
      return;
    }

    std::string bucket = bucket_it->second;
    std::string key = key_it->second;
    int64_t cluster_data_offset;

    try {
      cluster_data_offset = std::stoll(offset_it->second);
    } catch (...) {
      SendError("INVALID_PARAM", "Invalid cluster_data_offset");
      return;
    }

    std::cout << "[Client " << socket_fd_ << "] Loading index: s3://" << bucket
              << "/" << key << " (offset=" << cluster_data_offset << ")"
              << std::endl;

    // Check if this index is already loaded
    int existing_id = server_state_->FindIndexByLocation(
        bucket, key, cluster_data_offset);

    if (existing_id != -1) {
      std::cout << "[Client " << socket_fd_
                << "] Index already loaded with ID=" << existing_id
                << ", returning existing" << std::endl;

      // Get the existing index to return metadata
      IndexState *existing_state;
      if (server_state_->GetIndex(existing_id, &existing_state)) {
        auto *ivf_index =
            dynamic_cast<faiss::IndexIVFFlat *>(existing_state->index_.get());

        std::map<std::string, std::string> response;
        response["index"] = std::to_string(existing_id);
        response["d"] = std::to_string(ivf_index->d);
        response["ntotal"] = std::to_string(ivf_index->ntotal);
        response["nlist"] = std::to_string(ivf_index->nlist);
        response["metric_type"] = std::to_string(ivf_index->metric_type);

        SendResponse(ProtocolParser::FormatResponse(response));
        return;
      }
    }

    try {
      // Create S3 client
      auto s3_client = CreateS3Client();

      // Download index metadata (everything before cluster data)
      std::cout << "[Client " << socket_fd_ << "] Downloading metadata (0-"
                << cluster_data_offset - 1 << ")" << std::endl;

      auto metadata_bytes = DownloadRangeFromS3(
          s3_client, bucket, key, 0, cluster_data_offset);

      // Parse index with S3 flag (skips cluster data)
      faiss::VectorIOReader reader;
      reader.data = std::move(metadata_bytes);

      faiss::Index *raw_index = faiss::read_index(&reader, IO_FLAG_S3);

      // Cast to IVF index
      auto *ivf_index = dynamic_cast<faiss::IndexIVFFlat *>(raw_index);
      if (!ivf_index) {
        delete raw_index;
        SendError("LOAD_FAILED", "Index is not IndexIVFFlat type");
        return;
      }

      std::cout << "[Client " << socket_fd_
                << "] Parsed index: d=" << ivf_index->d
                << ", ntotal=" << ivf_index->ntotal
                << ", nlist=" << ivf_index->nlist << std::endl;

      // Extract placeholder inverted lists created by IO hook
      auto *placeholder =
          dynamic_cast<S3ReadNothingInvertedLists *>(ivf_index->invlists);

      if (!placeholder) {
        delete raw_index;
        SendError("LOAD_FAILED", "Invalid inverted lists type");
        return;
      }

      // Create S3OnDemandInvertedLists to replace placeholder
      auto *s3_invlists = new S3OnDemandInvertedLists(
          s3_client, bucket, key, cluster_data_offset, ivf_index->nlist,
          ivf_index->code_size, placeholder->cluster_sizes);

      // Configure cache size from environment variable
      const char *cache_size_env = std::getenv(kEnvCacheSizeMB);
      size_t cache_size_mb = kDefaultCacheSizeMB;
      if (cache_size_env) {
        try {
          cache_size_mb = std::stoull(cache_size_env);
          std::cout << "[Server] Using cache size from env: " << cache_size_mb
                    << " MB" << std::endl;
        } catch (...) {
          std::cerr << "[Server] Invalid " << kEnvCacheSizeMB
                    << " value, using default: " << cache_size_mb << " MB"
                    << std::endl;
        }
      } else {
        std::cout << "[Server] Using default cache size: " << cache_size_mb
                  << " MB (set " << kEnvCacheSizeMB << " to override)"
                  << std::endl;
      }

      // Set cache limit (0 = unlimited)
      if (cache_size_mb > kUnlimitedCache) {
        s3_invlists->SetMaxCacheBytes(cache_size_mb * kBytesPerMB);
      }

      // Replace placeholder with on-demand inverted lists
      ivf_index->replace_invlists(s3_invlists, true); // owns=true

      // Store in server state
      std::shared_ptr<faiss::Index> index_ptr(raw_index);
      int index_id = server_state_->AddIndex(
          bucket, key, cluster_data_offset, index_ptr, s3_client, s3_invlists);

      std::cout << "[Client " << socket_fd_
                << "] Index loaded with ID=" << index_id << std::endl;

      // Send response with metadata
      std::map<std::string, std::string> response;
      response["index"] = std::to_string(index_id);
      response["d"] = std::to_string(ivf_index->d);
      response["ntotal"] = std::to_string(ivf_index->ntotal);
      response["nlist"] = std::to_string(ivf_index->nlist);
      response["metric_type"] = std::to_string(ivf_index->metric_type);

      SendResponse(ProtocolParser::FormatResponse(response));

    } catch (const std::exception &e) {
      SendError("LOAD_FAILED",
                std::string("Failed to load index: ") + e.what());
    }
  }

  /**
   * Handles the SEARCH command.
   *
   * Performs k-NN search on a loaded index. The query vector is sent as
   * binary data following the command line.
   *
   * Command: SEARCH index=<id> k=<k> d=<dimension>
   * Binary input: [4-byte length][float array of size d]
   * Response: k=<k>
   * Binary output: [4-byte length][int64 array of IDs]
   *                [4-byte length][float32 array of distances]
   */
  void HandleSearch(const std::map<std::string, std::string> &params) {
    // Parse parameters
    auto index_it = params.find("index");
    auto k_it = params.find("k");
    auto d_it = params.find("d");

    if (index_it == params.end() || k_it == params.end() ||
        d_it == params.end()) {
      SendError("MISSING_PARAM", "Missing required parameters (index, k, d)");
      return;
    }

    int index_id, k, d;
    try {
      index_id = std::stoi(index_it->second);
      k = std::stoi(k_it->second);
      d = std::stoi(d_it->second);
    } catch (...) {
      SendError("INVALID_PARAM", "Invalid parameter values");
      return;
    }

    // Read query vector (binary) - MUST read before validation
    std::vector<uint8_t> query_data;
    if (!BinaryIO::ReadBinaryArray(socket_fd_, query_data)) {
      SendError("INVALID_BINARY", "Failed to read query vector");
      return;
    }

    // Validate query vector size
    size_t expected_size = d * sizeof(float);
    if (query_data.size() != expected_size) {
      SendError("INVALID_BINARY",
                "Expected " + std::to_string(expected_size) + " bytes, got " +
                    std::to_string(query_data.size()));
      return;
    }

    // Get index from server state
    IndexState *index_state;
    if (!server_state_->GetIndex(index_id, &index_state)) {
      SendError("INDEX_NOT_FOUND",
                "Index " + std::to_string(index_id) + " not loaded");
      return;
    }

    // Cast to IVF index
    auto *ivf_index =
        dynamic_cast<faiss::IndexIVFFlat *>(index_state->index_.get());

    if (!ivf_index) {
      SendError("INTERNAL_ERROR", "Index is not IVF type");
      return;
    }

    // Validate dimension
    if (ivf_index->d != static_cast<size_t>(d)) {
      SendError("INVALID_PARAM",
                "Dimension mismatch: index has d=" +
                    std::to_string(ivf_index->d) +
                    ", query has d=" + std::to_string(d));
      return;
    }

    std::cout << "[Client " << socket_fd_ << "] Search: index=" << index_id
              << ", k=" << k << ", d=" << d << std::endl;

    try {
      // Perform Faiss search
      const float *query = reinterpret_cast<const float *>(query_data.data());

      std::vector<float> distances(k);
      std::vector<faiss::idx_t> labels(k);

      // This triggers on-demand S3 fetching
      ivf_index->search(1, query, k, distances.data(), labels.data());

      std::cout << "[Client " << socket_fd_
                << "] Search completed: cache_hits="
                << index_state->s3_invlists_->CacheHits()
                << ", cache_misses="
                << index_state->s3_invlists_->CacheMisses() << std::endl;

      // Send text response
      std::map<std::string, std::string> response;
      response["k"] = std::to_string(k);
      if (!SendResponse(ProtocolParser::FormatResponse(response))) {
        return;
      }

      // Send binary data: IDs (int64)
      if (!BinaryIO::WriteBinaryArray(socket_fd_, labels.data(),
                                      k * sizeof(faiss::idx_t))) {
        std::cout << "[Client " << socket_fd_ << "] Failed to send result IDs"
                  << std::endl;
        return;
      }

      // Send binary data: distances (float32)
      if (!BinaryIO::WriteBinaryArray(socket_fd_, distances.data(),
                                      k * sizeof(float))) {
        std::cout << "[Client " << socket_fd_ << "] Failed to send distances"
                  << std::endl;
        return;
      }

      std::cout << "[Client " << socket_fd_ << "] Results sent successfully"
                << std::endl;

    } catch (const std::exception &e) {
      SendError("SEARCH_FAILED", std::string("Search failed: ") + e.what());
    }
  }

  /**
   * Handles the INFO command.
   *
   * Returns statistics about the cache or a specific index.
   *
   * Command: INFO about=cache
   * Response: index_count=<n> cache_hits=<hits> cache_misses=<misses>
   *
   * Command: INFO about=index id=<id>
   * Response: cluster_count=<n> cache_hits=<hits> cache_misses=<misses>
   *           cached_clusters=<n> cache_bytes=<bytes> cache_mb=<mb>
   *           max_cache_mb=<mb> nprobe=<n>
   */
  void HandleInfo(const std::map<std::string, std::string> &params) {
    auto about_it = params.find("about");
    if (about_it == params.end()) {
      SendError("MISSING_PARAM", "Missing about parameter");
      return;
    }

    std::map<std::string, std::string> response;

    if (about_it->second == "cache") {
      // Global cache statistics
      int index_count = server_state_->GetIndexCount();

      // Aggregate statistics from all indexes
      int64_t total_hits = 0;
      int64_t total_misses = 0;

      {
        std::lock_guard<std::mutex> lock(server_state_->GetIndexesMutex());
        for (const auto &kv : server_state_->GetLoadedIndexes()) {
          const IndexState &state = kv.second;
          if (state.s3_invlists_) {
            total_hits += state.s3_invlists_->CacheHits();
            total_misses += state.s3_invlists_->CacheMisses();
          }
        }
      }

      response["index_count"] = std::to_string(index_count);
      response["cache_hits"] = std::to_string(total_hits);
      response["cache_misses"] = std::to_string(total_misses);
      SendResponse(ProtocolParser::FormatResponse(response));

    } else if (about_it->second == "index") {
      // Per-index statistics
      auto id_it = params.find("id");
      if (id_it == params.end()) {
        SendError("MISSING_PARAM", "Missing id parameter for index query");
        return;
      }

      int index_id;
      try {
        index_id = std::stoi(id_it->second);
      } catch (...) {
        SendError("INVALID_PARAM", "Invalid id parameter");
        return;
      }

      IndexState *index_state;
      if (!server_state_->GetIndex(index_id, &index_state)) {
        SendError("INDEX_NOT_FOUND",
                  "Index " + std::to_string(index_id) + " not loaded");
        return;
      }

      auto *ivf = dynamic_cast<faiss::IndexIVFFlat *>(index_state->index_.get());

      response["cluster_count"] = std::to_string(ivf->nlist);
      response["cache_hits"] =
          std::to_string(index_state->s3_invlists_->CacheHits());
      response["cache_misses"] =
          std::to_string(index_state->s3_invlists_->CacheMisses());
      response["cached_clusters"] =
          std::to_string(index_state->s3_invlists_->CacheSize());
      response["cache_bytes"] =
          std::to_string(index_state->s3_invlists_->CacheBytes());
      response["cache_mb"] = std::to_string(
          index_state->s3_invlists_->CacheBytes() / kBytesPerMB);
      response["max_cache_mb"] = std::to_string(
          index_state->s3_invlists_->GetMaxCacheBytes() / kBytesPerMB);
      response["nprobe"] = std::to_string(ivf->nprobe);

      SendResponse(ProtocolParser::FormatResponse(response));

    } else {
      SendError("INVALID_PARAM", "Invalid about value: " + about_it->second);
    }
  }
};

} // namespace faiss_s3
