#pragma once

#include <map>
#include <sstream>
#include <string>

namespace faiss_s3 {

/**
 * Protocol utilities for the Faiss S3 cache server.
 *
 * The protocol uses space-separated key=value pairs followed by newline:
 *   COMMAND key1=value1 key2=value2\n
 *
 * Binary data follows the text protocol with 4-byte little-endian length prefix.
 *
 * Error responses start with "ERROR":
 *   ERROR code=ERROR_CODE msg=description\n
 */
class ProtocolParser {
public:
  /**
   * Parses a command line into a map of parameters.
   *
   * The first token is stored under the special key "__command".
   * Subsequent tokens are parsed as key=value pairs.
   *
   * Example:
   *   "LOAD bucket=my-bucket key=index.faiss"
   *   -> { "__command": "LOAD", "bucket": "my-bucket", "key": "index.faiss" }
   *
   * @param line Command line to parse (without trailing newline)
   * @return Map of parameter keys to values
   */
  static std::map<std::string, std::string>
  ParseCommand(const std::string &line) {
    std::map<std::string, std::string> params;
    std::istringstream iss(line);
    std::string token;

    // First token is the command
    if (iss >> token) {
      params["__command"] = token;
    }

    // Parse key=value pairs
    while (iss >> token) {
      size_t eq_pos = token.find('=');
      if (eq_pos != std::string::npos) {
        std::string key = token.substr(0, eq_pos);
        std::string value = token.substr(eq_pos + 1);
        params[key] = value;
      }
    }

    return params;
  }

  /**
   * Formats a response map into a protocol string.
   *
   * Parameters are joined with spaces and terminated with newline.
   *
   * Example:
   *   { "status": "ok", "index": "1" }
   *   -> "status=ok index=1\n"
   *
   * @param params Map of response parameters
   * @return Formatted protocol string with trailing newline
   */
  static std::string
  FormatResponse(const std::map<std::string, std::string> &params) {
    std::ostringstream oss;
    bool first = true;
    for (const auto &kv : params) {
      if (!first) {
        oss << " ";
      }
      oss << kv.first << "=" << kv.second;
      first = false;
    }
    oss << "\n";
    return oss.str();
  }

  /**
   * Formats an error response.
   *
   * Creates an ERROR response with code and message parameters.
   *
   * Example:
   *   FormatError("INVALID_PARAM", "Missing bucket")
   *   -> "ERROR code=INVALID_PARAM msg=Missing bucket\n"
   *
   * @param code Error code (e.g., "INVALID_PARAM", "INDEX_NOT_FOUND")
   * @param msg Human-readable error message
   * @return Formatted error string with trailing newline
   */
  static std::string FormatError(const std::string &code,
                                  const std::string &msg) {
    return "ERROR code=" + code + " msg=" + msg + "\n";
  }
};

} // namespace faiss_s3
