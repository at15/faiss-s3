#pragma once

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <iostream>
#include <thread>
#include <vector>

#include "client_handler.h"
#include "constants.h"
#include "server_state.h"

namespace faiss_s3 {

/**
 * TCP server for handling Faiss S3 cache requests.
 *
 * This server listens on a TCP port and spawns a new thread for each
 * client connection. Each thread runs a ClientHandler to process commands.
 *
 * The server supports graceful shutdown via the Stop() method, which
 * closes the listening socket and waits for all client threads to finish.
 */
class TCPServer {
public:
  /**
   * Creates a TCP server that will listen on the specified port.
   *
   * @param port TCP port number to listen on
   */
  explicit TCPServer(int port)
      : server_socket_(-1), port_(port), running_(false) {}

  /**
   * Destructor stops the server if still running.
   */
  ~TCPServer() { Stop(); }

  // Delete copy/move constructors (server is non-copyable)
  TCPServer(const TCPServer &) = delete;
  TCPServer &operator=(const TCPServer &) = delete;
  TCPServer(TCPServer &&) = delete;
  TCPServer &operator=(TCPServer &&) = delete;

  /**
   * Starts the TCP server.
   *
   * Creates the listening socket, binds to the port, and begins listening
   * for connections. Does not block - use Run() to enter the accept loop.
   *
   * @return true if server started successfully, false on error
   */
  bool Start() {
    // Create socket
    server_socket_ = socket(AF_INET, SOCK_STREAM, 0);
    if (server_socket_ < 0) {
      std::cerr << "[Server] Failed to create socket" << std::endl;
      return false;
    }

    // Set socket options (allow address reuse)
    int opt = 1;
    if (setsockopt(server_socket_, SOL_SOCKET, SO_REUSEADDR, &opt,
                   sizeof(opt)) < 0) {
      std::cerr << "[Server] Failed to set socket options" << std::endl;
      close(server_socket_);
      return false;
    }

    // Bind to port
    struct sockaddr_in address;
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(port_);

    if (bind(server_socket_, (struct sockaddr *)&address, sizeof(address)) <
        0) {
      std::cerr << "[Server] Failed to bind to port " << port_ << std::endl;
      close(server_socket_);
      return false;
    }

    // Listen for connections
    if (listen(server_socket_, kListenBacklog) < 0) {
      std::cerr << "[Server] Failed to listen on port " << port_ << std::endl;
      close(server_socket_);
      return false;
    }

    running_ = true;
    std::cout << "[Server] Listening on port " << port_ << std::endl;
    return true;
  }

  /**
   * Main server loop: accepts connections and spawns handler threads.
   *
   * This function blocks until Stop() is called or an error occurs.
   * Each accepted connection spawns a new thread running a ClientHandler.
   *
   * Note: Client threads are accumulated in a vector and never cleaned up
   * until server shutdown. For production use, consider using a thread pool
   * or detaching threads.
   */
  void Run() {
    while (running_) {
      // Accept client connection
      struct sockaddr_in client_address;
      socklen_t client_len = sizeof(client_address);

      int client_socket =
          accept(server_socket_, (struct sockaddr *)&client_address, &client_len);

      if (client_socket < 0) {
        if (running_) {
          std::cerr << "[Server] Failed to accept connection" << std::endl;
        }
        continue; // Server was stopped, accept() was interrupted
      }

      // Spawn thread for client
      // Note: Each client gets its own thread. Consider using a thread pool
      // for better resource management in production.
      client_threads_.emplace_back([this, client_socket]() {
        ClientHandler handler(client_socket, &server_state_);
        handler.Run();
      });
    }
  }

  /**
   * Stops the server and waits for all client threads to finish.
   *
   * This method:
   * 1. Sets the running flag to false
   * 2. Closes the listening socket (interrupts accept())
   * 3. Waits for all client handler threads to complete
   *
   * This is safe to call multiple times.
   */
  void Stop() {
    if (!running_) {
      return; // Already stopped
    }

    running_ = false;

    // Close server socket (this will interrupt accept())
    if (server_socket_ >= 0) {
      close(server_socket_);
      server_socket_ = -1;
    }

    // Wait for all client threads to finish
    // Note: This can take a while if clients are still connected
    std::cout << "[Server] Waiting for " << client_threads_.size()
              << " client threads to finish..." << std::endl;

    for (auto &thread : client_threads_) {
      if (thread.joinable()) {
        thread.join();
      }
    }

    std::cout << "[Server] Stopped" << std::endl;
  }

  /**
   * Returns whether the server is currently running.
   *
   * @return true if server is running, false otherwise
   */
  bool IsRunning() const { return running_; }

private:
  int server_socket_;
  int port_;
  std::atomic<bool> running_;
  ServerState server_state_;
  std::vector<std::thread> client_threads_;
};

} // namespace faiss_s3
