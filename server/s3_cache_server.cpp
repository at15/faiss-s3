/**
 * Faiss S3 Cache Server
 *
 * A TCP server that enables on-demand Faiss vector search from S3.
 * Loads index metadata on startup and fetches cluster data lazily during
 * searches with LRU caching.
 *
 * Protocol:
 *   - Text-based commands with key=value parameters
 *   - Binary data with 4-byte little-endian length prefix
 *   - See protocol.h for details
 *
 * Commands:
 *   - ECHO: Test connection
 *   - LOAD: Load index from S3
 *   - SEARCH: Perform k-NN search
 *   - INFO: Get cache statistics
 *
 * Usage:
 *   s3_cache_server [port]
 *
 * Environment variables:
 *   - S3_ENDPOINT_URL: Custom S3 endpoint (for S3Mock, MinIO)
 *   - AWS_REGION: AWS region (default: us-east-1)
 *   - FAISS_S3_CACHE_SIZE_MB: Cache size in MB (default: 2048)
 */

#include <csignal>
#include <cstdlib>
#include <iostream>
#include <string>

#include "constants.h"
#include "tcp_server.h"

// Global server pointer for signal handling
faiss_s3::TCPServer *g_server = nullptr;

/**
 * Signal handler for graceful shutdown (SIGINT, SIGTERM).
 *
 * Stops the server when receiving interrupt or termination signals.
 */
void SignalHandler(int signal) {
  std::cout << "\n[Server] Received signal " << signal
            << ", shutting down..." << std::endl;
  if (g_server) {
    g_server->Stop();
  }
}

/**
 * Main entry point.
 *
 * Parses command-line arguments, starts the TCP server, and enters
 * the main accept loop. Handles graceful shutdown on signals.
 */
int main(int argc, char *argv[]) {
  int port = faiss_s3::kDefaultServerPort;

  // Parse command-line arguments
  if (argc > 1) {
    try {
      port = std::stoi(argv[1]);
    } catch (...) {
      std::cerr << "Invalid port number: " << argv[1] << std::endl;
      std::cerr << "Usage: " << argv[0] << " [port]" << std::endl;
      return 1;
    }
  }

  // Create server
  faiss_s3::TCPServer server(port);
  g_server = &server;

  // Register signal handlers for graceful shutdown
  signal(SIGINT, SignalHandler);
  signal(SIGTERM, SignalHandler);

  // Start server
  if (!server.Start()) {
    std::cerr << "[Server] Failed to start server" << std::endl;
    return 1;
  }

  // Run server (blocks until Stop() is called)
  server.Run();

  return 0;
}
