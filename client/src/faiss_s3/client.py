#!/usr/bin/env python3
"""
S3 Cache Server Client

A Python client for communicating with the S3 Cache Server using a custom
TCP protocol. The server loads FAISS indexes from S3 and performs k-nearest
neighbor searches with on-demand cluster caching.

Protocol Details:
- Text commands: space-separated key=value pairs, newline-terminated
- Binary data: 4-byte little-endian length prefix + raw bytes
- Default port: 9001

Example usage:
    from faiss_s3.client import S3CacheClient

    with S3CacheClient(host="localhost", port=9001) as client:
        # Load index from S3
        index_id = client.load(
            bucket="my-bucket",
            key="my-index.ivf",
            cluster_data_offset=3154059
        )

        # Perform search
        import numpy as np
        query = np.random.randn(128).astype(np.float32)
        ids, distances = client.search(index_id, query, k=10)

        # Get statistics
        stats = client.info_cache()
        print(f"Cache hits: {stats['cache_hits']}")
"""

import socket
import struct
import numpy as np
from typing import Optional


class S3CacheClientError(Exception):
    """Exception raised for S3 cache client errors.

    Attributes:
        code: Error code returned by the server (e.g., 'INDEX_NOT_FOUND')
        msg: Human-readable error message
    """

    def __init__(self, code: str, msg: str):
        self.code = code
        self.msg = msg
        super().__init__(f"[{code}] {msg}")


class S3CacheClient:
    """Client for the S3 Cache Server.

    This client communicates with the S3 Cache Server using a custom TCP
    protocol for loading FAISS indexes from S3 and performing searches.

    Attributes:
        host: Server hostname or IP address
        port: Server port number
        socket: Active socket connection (None if not connected)
    """

    def __init__(self, host: str = "localhost", port: int = 9001):
        """Initialize the S3 Cache Client.

        Args:
            host: Server hostname or IP address (default: "localhost")
            port: Server port number (default: 9001)
        """
        self.host = host
        self.port = port
        self.socket: Optional[socket.socket] = None

    def connect(self):
        """Connect to the server.

        Establishes a TCP connection to the S3 Cache Server. If already
        connected, closes the existing connection first.

        Raises:
            ConnectionError: If connection fails
        """
        if self.socket is not None:
            self.close()

        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((self.host, self.port))

    def close(self):
        """Close the connection to the server."""
        if self.socket:
            self.socket.close()
            self.socket = None

    def __enter__(self):
        """Context manager entry: connect to the server."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit: close the connection."""
        self.close()

    def _send_command(self, command: str, params: dict[str, str]):
        """Send a text command with parameters.

        Args:
            command: Command name (e.g., 'LOAD', 'SEARCH')
            params: Dictionary of parameter key-value pairs
        """
        if self.socket is None:
            raise RuntimeError("Not connected to server")
        parts = [command]
        for key, value in params.items():
            parts.append(f"{key}={value}")
        command_line = " ".join(parts) + "\n"
        self.socket.sendall(command_line.encode("utf-8"))

    def _send_binary(self, data: bytes):
        """Send binary data with 4-byte little-endian length prefix.

        Args:
            data: Binary data to send
        """
        if self.socket is None:
            raise RuntimeError("Not connected to server")
        length = len(data)
        self.socket.sendall(struct.pack("<I", length))  # 4-byte little-endian
        self.socket.sendall(data)

    def _recv_exact(self, length: int) -> bytes:
        """Receive exactly the specified number of bytes.

        Args:
            length: Number of bytes to receive

        Returns:
            Received bytes

        Raises:
            ConnectionError: If connection closed before all bytes received
        """
        if self.socket is None:
            raise RuntimeError("Not connected to server")
        data = b""
        while len(data) < length:
            chunk = self.socket.recv(length - len(data))
            if not chunk:
                raise ConnectionError("Connection closed while reading data")
            data += chunk
        return data

    def _recv_line(self) -> str:
        """Receive a line of text (until newline).

        Returns:
            Received line as string (without newline)

        Raises:
            ConnectionError: If connection closed while reading
            ValueError: If line exceeds maximum length (8192 bytes)
        """
        if self.socket is None:
            raise RuntimeError("Not connected to server")
        line = b""
        while True:
            char = self.socket.recv(1)
            if not char:
                raise ConnectionError("Connection closed while reading line")
            if char == b"\n":
                break
            line += char
            if len(line) > 8192:  # Max line length
                raise ValueError("Line too long")
        return line.decode("utf-8")

    def _recv_binary(self) -> bytes:
        """Receive binary data with 4-byte little-endian length prefix.

        Returns:
            Received binary data
        """
        length_bytes = self._recv_exact(4)
        length = struct.unpack("<I", length_bytes)[0]
        return self._recv_exact(length)

    def _parse_response(self, line: str) -> dict[str, str]:
        """Parse a response line into key=value pairs.

        Args:
            line: Response line from server

        Returns:
            Dictionary of parsed key-value pairs

        Raises:
            S3CacheClientError: If response is an error
        """
        if line.startswith("ERROR "):
            # Parse error response
            params = {}
            for part in line[6:].split():
                if "=" in part:
                    key, value = part.split("=", 1)
                    params[key] = value

            code = params.get("code", "UNKNOWN")
            msg = params.get("msg", "Unknown error")
            raise S3CacheClientError(code, msg)

        # Parse normal response
        params = {}
        for part in line.split():
            if "=" in part:
                key, value = part.split("=", 1)
                params[key] = value
        return params

    # Command methods

    def echo(self, msg: str) -> str:
        """Test connection with ECHO command.

        Args:
            msg: Message to echo

        Returns:
            Echoed message from server

        Raises:
            S3CacheClientError: If server returns an error
        """
        self._send_command("ECHO", {"msg": msg})
        response_line = self._recv_line()
        response = self._parse_response(response_line)
        return response.get("msg", "")

    def load(self, bucket: str, key: str, cluster_data_offset: int) -> int:
        """Load a FAISS index from S3.

        The server downloads the index metadata and sets up on-demand loading
        for cluster data. If the same index is already loaded, returns the
        existing index ID.

        Args:
            bucket: S3 bucket name
            key: S3 object key (path to index file)
            cluster_data_offset: Byte offset where cluster data begins

        Returns:
            Server-assigned index ID (used for subsequent operations)

        Raises:
            S3CacheClientError: If loading fails (e.g., LOAD_FAILED)
        """
        params = {
            "bucket": bucket,
            "key": key,
            "cluster_data_offset": str(cluster_data_offset),
        }
        self._send_command("LOAD", params)
        response_line = self._recv_line()
        response = self._parse_response(response_line)
        return int(response["index"])

    def search(
        self, index_id: int, query: np.ndarray, k: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Search an index for k nearest neighbors.

        Args:
            index_id: Index ID returned by load()
            query: Query vector as numpy array (will be converted to float32)
            k: Number of nearest neighbors to return

        Returns:
            Tuple of (ids, distances) as numpy arrays:
                - ids: int64 array of shape (k,) containing vector IDs
                - distances: float32 array of shape (k,) containing distances/scores

        Raises:
            S3CacheClientError: If search fails (e.g., INDEX_NOT_FOUND, INVALID_PARAM)
        """
        # Validate query vector
        if query.dtype != np.float32:
            query = query.astype(np.float32)

        d = len(query)

        # Send command with binary data
        params = {"index": str(index_id), "k": str(k), "d": str(d)}
        self._send_command("SEARCH", params)

        # Send query vector (binary) immediately after command
        query_bytes = query.tobytes()
        self._send_binary(query_bytes)

        # Receive response
        response_line = self._recv_line()
        response = self._parse_response(response_line)
        result_k = int(response["k"])

        # Receive IDs (binary)
        ids_bytes = self._recv_binary()
        ids = np.frombuffer(ids_bytes, dtype=np.int64)

        # Receive distances (binary)
        distances_bytes = self._recv_binary()
        distances = np.frombuffer(distances_bytes, dtype=np.float32)

        # Validate sizes
        if len(ids) != result_k or len(distances) != result_k:
            raise ValueError(
                f"Expected {result_k} results, got {len(ids)} ids and {len(distances)} distances"
            )

        return ids, distances

    def info_cache(self) -> dict[str, int]:
        """Get global cache statistics.

        Returns:
            Dictionary with keys:
                - index_count: Number of loaded indexes
                - cache_hits: Total cache hits across all indexes
                - cache_misses: Total cache misses across all indexes

        Raises:
            S3CacheClientError: If query fails
        """
        self._send_command("INFO", {"about": "cache"})
        response_line = self._recv_line()
        response = self._parse_response(response_line)

        return {
            "index_count": int(response.get("index_count", 0)),
            "cache_hits": int(response.get("cache_hits", 0)),
            "cache_misses": int(response.get("cache_misses", 0)),
        }

    def info_index(self, index_id: int) -> dict[str, int]:
        """Get per-index statistics.

        Args:
            index_id: Index ID to query

        Returns:
            Dictionary with keys:
                - cluster_count: Total number of clusters
                - cache_hits: Cache hits for this index
                - cache_misses: Cache misses for this index
                - cached_clusters: Number of currently cached clusters
                - nprobe: Number of clusters searched per query (if available)

        Raises:
            S3CacheClientError: If query fails (e.g., INDEX_NOT_FOUND)
        """
        params = {"about": "index", "id": str(index_id)}
        self._send_command("INFO", params)
        response_line = self._recv_line()
        response = self._parse_response(response_line)

        result = {
            "cluster_count": int(response.get("cluster_count", 0)),
            "cache_hits": int(response.get("cache_hits", 0)),
            "cache_misses": int(response.get("cache_misses", 0)),
            "cached_clusters": int(response.get("cached_clusters", 0)),
        }

        # nprobe is optional
        if "nprobe" in response:
            result["nprobe"] = int(response["nprobe"])

        return result
