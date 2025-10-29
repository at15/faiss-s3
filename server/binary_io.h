#pragma once

#include <cstdint>
#include <iostream>
#include <sys/socket.h>
#include <vector>

#include "constants.h"

namespace faiss_s3 {

/**
 * Binary I/O utilities for socket communication.
 *
 * Provides helper functions for reading and writing binary data over sockets
 * with proper length prefixes and bounds checking.
 */
class BinaryIO {
public:
  /**
   * Reads exactly the specified number of bytes from a socket.
   *
   * This function handles partial reads by looping until all bytes are read
   * or an error occurs.
   *
   * @param socket_fd Socket file descriptor
   * @param buffer Destination buffer (must be at least 'length' bytes)
   * @param length Number of bytes to read
   * @return true if all bytes were read successfully, false on error or EOF
   */
  static bool ReadExact(int socket_fd, void *buffer, size_t length) {
    size_t total_read = 0;
    uint8_t *buf = static_cast<uint8_t *>(buffer);

    while (total_read < length) {
      ssize_t n = recv(socket_fd, buf + total_read, length - total_read, 0);
      if (n <= 0) {
        return false; // Connection closed or error
      }
      total_read += n;
    }

    return true;
  }

  /**
   * Writes exactly the specified number of bytes to a socket.
   *
   * This function handles partial writes by looping until all bytes are written
   * or an error occurs.
   *
   * @param socket_fd Socket file descriptor
   * @param buffer Source buffer containing data to write
   * @param length Number of bytes to write
   * @return true if all bytes were written successfully, false on error
   */
  static bool WriteExact(int socket_fd, const void *buffer, size_t length) {
    size_t total_written = 0;
    const uint8_t *buf = static_cast<const uint8_t *>(buffer);

    while (total_written < length) {
      ssize_t n = send(socket_fd, buf + total_written, length - total_written, 0);
      if (n <= 0) {
        return false; // Connection closed or error
      }
      total_written += n;
    }

    return true;
  }

  /**
   * Reads a length-prefixed binary array from a socket.
   *
   * Protocol format:
   *   [4 bytes: uint32_t length in little-endian]
   *   [length bytes: data]
   *
   * The function includes protection against memory exhaustion attacks by
   * limiting the maximum array size to MAX_BINARY_ARRAY_SIZE.
   *
   * @param socket_fd Socket file descriptor
   * @param data Output vector that will be resized to contain the data
   * @return true if array was read successfully, false on error
   */
  static bool ReadBinaryArray(int socket_fd, std::vector<uint8_t> &data) {
    uint32_t length;
    if (!ReadExact(socket_fd, &length, sizeof(length))) {
      return false;
    }

    // Validate array size to prevent OOM attacks
    if (length > MAX_BINARY_ARRAY_SIZE) {
      std::cerr << "[BinaryIO] Array size too large: " << length << " bytes (max "
                << MAX_BINARY_ARRAY_SIZE << " bytes)" << std::endl;
      return false;
    }

    data.resize(length);
    return ReadExact(socket_fd, data.data(), length);
  }

  /**
   * Writes a length-prefixed binary array to a socket.
   *
   * Protocol format:
   *   [4 bytes: uint32_t length in little-endian]
   *   [length bytes: data]
   *
   * @param socket_fd Socket file descriptor
   * @param data Pointer to data to write
   * @param length Number of bytes to write
   * @return true if array was written successfully, false on error
   */
  static bool WriteBinaryArray(int socket_fd, const void *data,
                                uint32_t length) {
    if (!WriteExact(socket_fd, &length, sizeof(length))) {
      return false;
    }
    return WriteExact(socket_fd, data, length);
  }

  /**
   * Reads a text line from a socket (up to newline character).
   *
   * This function reads one byte at a time until a newline is encountered.
   * The newline character is consumed but not included in the output.
   *
   * Note: This implementation is simple but inefficient (one syscall per byte).
   * Consider using buffered I/O for better performance if this becomes a
   * bottleneck.
   *
   * @param socket_fd Socket file descriptor
   * @param line Output string (newline not included)
   * @return true if line was read successfully, false on error or EOF
   */
  static bool ReadLine(int socket_fd, std::string &line) {
    line.clear();
    char ch;
    while (true) {
      ssize_t n = recv(socket_fd, &ch, 1, 0);
      if (n <= 0) {
        return false; // Connection closed or error
      }
      if (ch == '\n') {
        break; // End of line
      }
      if (line.size() >= MAX_COMMAND_LINE_LENGTH) {
        std::cerr << "[BinaryIO] Command line too long (max "
                  << MAX_COMMAND_LINE_LENGTH << " bytes)" << std::endl;
        return false;
      }
      line += ch;
    }
    return true;
  }
};

} // namespace faiss_s3
