#!/usr/bin/env python3
"""
Full End-to-End Test for faiss-s3

This test implements a complete workflow from raw Quora dataset to querying
via both local FAISS and S3 cache server, with result verification.

Test Flow:
1. Download and prepare Quora dataset
2. Generate embeddings using SentenceTransformers
3. Build FAISS IVF index with dynamic cluster calculation
4. Generate attributes.parquet (source text mapping)
5. Generate complete meta.json
6. Upload all files to S3 with unique prefix
7. Query via local FAISS (download + mmap)
8. Query via S3 cache server (on-demand loading)
9. Verify both methods return identical results
10. Report comprehensive benchmarks

Usage:
    python test_e2e_full.py [--dataset-size N] [--n-clusters N] [--k N] [--nprobe N]
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import boto3
import faiss
import numpy as np
import pandas as pd
from boto3.s3.transfer import TransferConfig

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    print("ERROR: pyarrow not installed")
    print("Please run: pip install pyarrow")
    sys.exit(1)

try:
    from sentence_transformers import SentenceTransformer, util
except ImportError:
    print("ERROR: sentence-transformers not installed")
    print("Please run: pip install sentence-transformers")
    sys.exit(1)

from faiss_s3 import S3CacheClient, generate_meta_from_file


class Timer:
    """Context manager for timing code blocks"""

    def __init__(self, name: str):
        self.name = name
        self.start_time = 0.0
        self.elapsed = 0.0

    def __enter__(self):
        print(f"\n[{self.name}] Starting...")
        self.start_time = time.monotonic()
        return self

    def __exit__(self, *args):
        self.elapsed = time.monotonic() - self.start_time
        print(f"[{self.name}] Completed in {self.elapsed:.2f}s")


class FullE2ETest:
    """Complete end-to-end test for faiss-s3"""

    def __init__(
        self,
        dataset_size: int = 10000,
        n_clusters: int | None = None,
        k: int = 5,
        nprobe: int = 10,
    ):
        # Configuration
        self.dataset_size = dataset_size
        self.k = k
        self.nprobe = nprobe

        # Auto-calculate n_clusters if not provided
        if n_clusters is None:
            # Use ~1% of dataset size, minimum 10
            self.n_clusters = max(10, dataset_size // 100)
        else:
            self.n_clusters = n_clusters

        # Ensure n_clusters < n_vectors
        if self.n_clusters >= self.dataset_size:
            raise ValueError(
                f"n_clusters ({self.n_clusters}) must be < dataset_size ({self.dataset_size})"
            )

        # Model configuration
        self.model_name = "quora-distilbert-multilingual"
        self.embedding_size = 768

        # S3 configuration
        self.s3_bucket = "test-bucket"
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.s3_prefix = f"quora-{timestamp}"

        # Temporary working directory
        self.work_dir = Path(f"./e2e_test_{timestamp}")
        self.work_dir.mkdir(exist_ok=True)

        # Shared cache directory for dataset (outside work_dir to persist across runs)
        self.cache_dir = Path("./e2e_cache")
        self.cache_dir.mkdir(exist_ok=True)

        # File paths
        self.dataset_path = self.cache_dir / "quora_duplicate_questions.tsv"
        self.embeddings_path = self.work_dir / "embeddings.npy"
        self.index_path = self.work_dir / "index.ivf"
        self.attributes_path = self.work_dir / "attributes.parquet"
        self.meta_path = self.work_dir / "meta.json"

        # Data storage
        self.corpus_sentences: list[str] = []
        self.corpus_embeddings: np.ndarray | None = None
        self.model: SentenceTransformer | None = None
        self.index: faiss.IndexIVFFlat | None = None

        # Benchmarking
        self.timings: dict[str, float] = {}

        print("\n" + "=" * 80)
        print("Full E2E Test Configuration")
        print("=" * 80)
        print(f"Dataset size: {self.dataset_size:,}")
        print(f"Number of clusters: {self.n_clusters}")
        print(f"Search k: {self.k}")
        print(f"Search nprobe: {self.nprobe}")
        print(f"Model: {self.model_name}")
        print(f"S3 bucket: {self.s3_bucket}")
        print(f"S3 prefix: {self.s3_prefix}")
        print(f"Cache directory: {self.cache_dir} (shared across runs)")
        print(f"Working directory: {self.work_dir}")

    def phase1_download_and_prepare_dataset(self):
        """Phase 1: Download and prepare Quora dataset"""
        with Timer("Phase 1: Download and Prepare Dataset") as t:
            url = "http://qim.fs.quoracdn.net/quora_duplicate_questions.tsv"

            if not self.dataset_path.exists():
                print(f"Downloading dataset from {url}")
                util.http_get(url, str(self.dataset_path))
                print(f"Dataset saved to {self.dataset_path}")
            else:
                print(f"Using cached dataset: {self.dataset_path}")

            # Load and extract unique questions
            df = pd.read_csv(self.dataset_path, sep="\t")
            print(f"Loaded {len(df):,} question pairs")

            corpus_sentences_set = set()
            for _, row in df.iterrows():
                corpus_sentences_set.add(row["question1"])
                if len(corpus_sentences_set) >= self.dataset_size:
                    break
                corpus_sentences_set.add(row["question2"])
                if len(corpus_sentences_set) >= self.dataset_size:
                    break

            self.corpus_sentences = list(corpus_sentences_set)
            print(f"Extracted {len(self.corpus_sentences):,} unique questions")

        self.timings["dataset_download"] = t.elapsed

    def phase2_generate_embeddings(self):
        """Phase 2: Generate embeddings using SentenceTransformers"""
        with Timer("Phase 2: Generate Embeddings") as t:
            print(f"Loading model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)

            print("Encoding corpus (this may take several minutes)...")
            self.corpus_embeddings = self.model.encode(
                self.corpus_sentences, show_progress_bar=True, convert_to_numpy=True
            )

            print(f"Embeddings shape: {self.corpus_embeddings.shape}")

            # Save embeddings to .npy file
            np.save(str(self.embeddings_path), self.corpus_embeddings)
            print(f"Embeddings saved to {self.embeddings_path}")

        self.timings["embedding_generation"] = t.elapsed

    def phase3_build_index(self):
        """Phase 3: Build FAISS IVF index"""
        with Timer("Phase 3: Build FAISS Index") as t:
            if self.corpus_embeddings is None:
                raise RuntimeError("Embeddings not generated")

            # Normalize embeddings for INNER_PRODUCT metric
            print("Normalizing embeddings...")
            normalized_embeddings = self.corpus_embeddings / np.linalg.norm(
                self.corpus_embeddings, axis=1
            )[:, None]

            # Create IVF index
            print(f"Creating IVF index with {self.n_clusters} clusters...")
            quantizer = faiss.IndexFlatIP(self.embedding_size)
            self.index = faiss.IndexIVFFlat(
                quantizer,
                self.embedding_size,
                self.n_clusters,
                faiss.METRIC_INNER_PRODUCT,
            )
            self.index.nprobe = self.nprobe

            # Train and add vectors
            print("Training index...")
            self.index.train(normalized_embeddings)

            print("Adding vectors to index...")
            self.index.add(normalized_embeddings)

            print(f"Index built: {self.index.ntotal} vectors in {self.n_clusters} clusters")

            # Save index
            faiss.write_index(self.index, str(self.index_path))
            print(f"Index saved to {self.index_path}")

        self.timings["index_building"] = t.elapsed

    def phase4_generate_attributes_parquet(self):
        """Phase 4: Generate attributes.parquet file"""
        with Timer("Phase 4: Generate Attributes Parquet") as t:
            # Create DataFrame with just text column
            # Row index implicitly equals vector ID
            df = pd.DataFrame({"text": self.corpus_sentences})

            print(f"Creating parquet file with {len(df)} rows")

            # Write to parquet using pyarrow
            table = pa.Table.from_pandas(df)
            pq.write_table(table, str(self.attributes_path))

            file_size = self.attributes_path.stat().st_size
            print(f"Attributes saved to {self.attributes_path} ({file_size:,} bytes)")

        self.timings["attributes_generation"] = t.elapsed

    def phase5_generate_meta_json(self):
        """Phase 5: Generate complete meta.json"""
        with Timer("Phase 5: Generate Meta JSON") as t:
            # Use generate_meta_from_file to get cluster_data_offset dynamically
            print("Calculating cluster_data_offset using generate_meta_from_file...")
            index_meta = generate_meta_from_file(str(self.index_path))
            cluster_data_offset = index_meta.cluster_data_offset
            print(f"Cluster data offset: {cluster_data_offset:,}")

            # Get file sizes
            index_size = self.index_path.stat().st_size
            attributes_size = self.attributes_path.stat().st_size
            embeddings_size = self.embeddings_path.stat().st_size

            # Build complete meta.json
            meta = {
                "version": "1",
                "index": {
                    "file": "index.ivf",
                    "size_bytes": index_size,
                    "type": "IndexIVFFlat",
                    "vector_size": self.embedding_size,
                    "n_vectors": len(self.corpus_sentences),
                    "n_clusters": self.n_clusters,
                    "cluster_data_offset": cluster_data_offset,
                },
                "attributes": {
                    "file": "attributes.parquet",
                    "size_bytes": attributes_size,
                    "columns": [{"name": "text", "type": "string"}],
                },
                "embedding": {
                    "file": "embeddings.npy",
                    "library": "sentence-transformers",
                    "model": self.model_name,
                    "vector_size": self.embedding_size,
                    "n_vectors": len(self.corpus_sentences),
                },
            }

            # Write meta.json
            with open(self.meta_path, "w") as f:
                json.dump(meta, f, indent=2)

            print(f"Meta.json saved to {self.meta_path}")
            print("Meta.json contents:")
            print(json.dumps(meta, indent=2))

        self.timings["meta_generation"] = t.elapsed

    def phase6_upload_to_s3(self):
        """Phase 6: Upload all files to S3"""
        with Timer("Phase 6: Upload to S3") as t:
            # Disable multipart for S3Mock compatibility
            config = TransferConfig(multipart_threshold=1024 * 1024 * 1024 * 100)  # 100GB
            s3 = boto3.client("s3")

            files_to_upload = [
                ("meta.json", self.meta_path),
                ("index.ivf", self.index_path),
                ("attributes.parquet", self.attributes_path),
                ("embeddings.npy", self.embeddings_path),
            ]

            for s3_key_suffix, local_path in files_to_upload:
                s3_key = f"{self.s3_prefix}/{s3_key_suffix}"
                print(f"Uploading {local_path.name} to s3://{self.s3_bucket}/{s3_key}")
                s3.upload_file(
                    str(local_path), self.s3_bucket, s3_key, Config=config
                )

            print(f"All files uploaded to s3://{self.s3_bucket}/{self.s3_prefix}/")

        self.timings["s3_upload"] = t.elapsed

    def phase7_validate_infrastructure(self):
        """Phase 7: Validate S3 cache server is running"""
        with Timer("Phase 7: Validate Infrastructure") as t:
            # S3Mock validation skipped - if upload in phase 6 succeeded, S3 is working

            # Check S3 cache server
            print("Checking S3 cache server...")
            try:
                with S3CacheClient(host="localhost", port=9001) as client:
                    response = client.echo("test")
                    if response == "test":
                        print("  ✓ S3 cache server is running")
                    else:
                        raise RuntimeError(f"Unexpected echo response: {response}")
            except Exception as e:
                print(f"\n✗ ERROR: S3 cache server not accessible on port 9001")
                print(f"  Error: {e}")
                print("\nSetup instructions:")
                print("  cd server && make docker-run")
                print("Or:")
                print("  cd server && make build && ./s3_cache_server")
                sys.exit(1)

        self.timings["infrastructure_validation"] = t.elapsed

    def phase8_query_local(self) -> tuple[np.ndarray, np.ndarray]:
        """Phase 8: Query using local FAISS (download + mmap)"""
        with Timer("Phase 8: Query via Local FAISS") as t:
            # Download files from S3
            print("Downloading files from S3...")
            s3 = boto3.client("s3")

            local_index = self.work_dir / "downloaded_index.ivf"
            local_meta = self.work_dir / "downloaded_meta.json"
            local_attributes = self.work_dir / "downloaded_attributes.parquet"

            s3.download_file(
                self.s3_bucket,
                f"{self.s3_prefix}/index.ivf",
                str(local_index),
            )
            s3.download_file(
                self.s3_bucket,
                f"{self.s3_prefix}/meta.json",
                str(local_meta),
            )
            s3.download_file(
                self.s3_bucket,
                f"{self.s3_prefix}/attributes.parquet",
                str(local_attributes),
            )

            print("Loading index locally...")
            local_faiss_index = faiss.read_index(str(local_index))

            # Generate query vector
            query_text = "How to find a job in technology"
            print(f'Query: "{query_text}"')

            if self.model is None:
                raise RuntimeError("Model not loaded")

            query_embedding = self.model.encode(query_text, convert_to_numpy=True)
            query_embedding = query_embedding / np.linalg.norm(query_embedding)
            query_embedding = np.expand_dims(query_embedding, axis=0)

            # Search
            print(f"Searching (k={self.k})...")
            distances, ids = local_faiss_index.search(query_embedding, self.k)

            # Load attributes for text mapping
            df_attributes = pd.read_parquet(str(local_attributes))

            print("\nLocal FAISS Results:")
            print("-" * 80)
            for rank, (vector_id, score) in enumerate(
                zip(ids[0], distances[0]), 1
            ):
                text = df_attributes.loc[vector_id, "text"]
                print(f"  {rank}. [id={vector_id}, score={score:.4f}] {text}")

        self.timings["local_query"] = t.elapsed
        return ids[0], distances[0]

    def phase9_query_server(
        self, query_vector: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Phase 9: Query using S3 cache server (on-demand loading)"""
        with Timer("Phase 9: Query via S3 Cache Server") as t:
            print("Connecting to S3 cache server...")

            # Load meta.json to get cluster_data_offset
            with open(self.meta_path) as f:
                meta = json.load(f)
            cluster_data_offset = meta["index"]["cluster_data_offset"]

            # Use same query as local method
            query_text = "How to find a job in technology"
            print(f'Query: "{query_text}"')

            if self.model is None:
                raise RuntimeError("Model not loaded")

            query_embedding = self.model.encode(query_text, convert_to_numpy=True)
            query_embedding = query_embedding / np.linalg.norm(query_embedding)
            query_embedding = query_embedding.astype(np.float32)

            # Load attributes for text mapping
            df_attributes = pd.read_parquet(str(self.attributes_path))

            with S3CacheClient(host="localhost", port=9001) as client:
                # Load index from S3
                print(f"Loading index from S3 (bucket={self.s3_bucket}, key={self.s3_prefix}/index.ivf)")
                index_id = client.load(
                    self.s3_bucket,
                    f"{self.s3_prefix}/index.ivf",
                    cluster_data_offset,
                )
                print(f"Index loaded with ID: {index_id}")

                # Search
                print(f"Searching (k={self.k})...")
                ids, distances = client.search(index_id, query_embedding, self.k)

                print("\nS3 Cache Server Results:")
                print("-" * 80)
                for rank, (vector_id, score) in enumerate(zip(ids, distances), 1):
                    text = df_attributes.loc[vector_id, "text"]
                    print(f"  {rank}. [id={vector_id}, score={score:.4f}] {text}")

                # Get cache statistics
                print("\nCache Statistics:")
                cache_stats = client.info_cache()
                index_stats = client.info_index(index_id)

                print(f"  Global cache hits: {cache_stats['cache_hits']:,}")
                print(f"  Global cache misses: {cache_stats['cache_misses']:,}")
                print(f"  Index cached clusters: {index_stats['cached_clusters']}/{index_stats['cluster_count']}")
                print(f"  Index cache hits: {index_stats['cache_hits']:,}")
                print(f"  Index cache misses: {index_stats['cache_misses']:,}")

                if index_stats["cache_hits"] + index_stats["cache_misses"] > 0:
                    hit_rate = (
                        index_stats["cache_hits"]
                        / (index_stats["cache_hits"] + index_stats["cache_misses"])
                        * 100
                    )
                    print(f"  Cache hit rate: {hit_rate:.1f}%")

        self.timings["server_query"] = t.elapsed
        return ids, distances

    def phase10_verify_results(
        self,
        local_ids: np.ndarray,
        server_ids: np.ndarray,
    ):
        """Phase 10: Verify both query methods return identical results"""
        with Timer("Phase 10: Verify Results") as t:
            print("\nComparing results...")

            local_id_set = set(local_ids.tolist())
            server_id_set = set(server_ids.tolist())

            print(f"Local IDs: {sorted(local_id_set)}")
            print(f"Server IDs: {sorted(server_id_set)}")

            if local_id_set == server_id_set:
                print("✓ SUCCESS: Both methods returned identical vector IDs")
            else:
                symmetric_diff = local_id_set ^ server_id_set
                print(f"✗ FAILURE: Result mismatch!")
                print(f"  Symmetric difference: {symmetric_diff}")
                print(f"  Only in local: {local_id_set - server_id_set}")
                print(f"  Only in server: {server_id_set - local_id_set}")
                sys.exit(1)

        self.timings["result_verification"] = t.elapsed

    def phase11_report_benchmarks(self):
        """Phase 11: Report comprehensive benchmarks"""
        print("\n" + "=" * 80)
        print("BENCHMARK SUMMARY")
        print("=" * 80)

        total_time = sum(self.timings.values())

        for phase, elapsed in self.timings.items():
            percentage = (elapsed / total_time * 100) if total_time > 0 else 0
            print(f"  {phase:30s}: {elapsed:8.2f}s ({percentage:5.1f}%)")

        print(f"\n  {'TOTAL':30s}: {total_time:8.2f}s")

        print("\n" + "=" * 80)
        print("TEST COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print(f"S3 location: s3://{self.s3_bucket}/{self.s3_prefix}/")
        print(f"Working directory: {self.work_dir}")

    def run(self):
        """Run the complete end-to-end test"""
        try:
            self.phase1_download_and_prepare_dataset()
            self.phase2_generate_embeddings()
            self.phase3_build_index()
            self.phase4_generate_attributes_parquet()
            self.phase5_generate_meta_json()
            self.phase6_upload_to_s3()
            self.phase7_validate_infrastructure()
            local_ids, local_distances = self.phase8_query_local()
            server_ids, server_distances = self.phase9_query_server(
                np.zeros((1, self.embedding_size), dtype=np.float32)
            )
            self.phase10_verify_results(local_ids, server_ids)
            self.phase11_report_benchmarks()

        except KeyboardInterrupt:
            print("\n\nTest interrupted by user")
            sys.exit(1)
        except Exception as e:
            print(f"\n\nFATAL ERROR: {e}")
            import traceback

            traceback.print_exc()
            sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Full end-to-end test for faiss-s3"
    )
    parser.add_argument(
        "--dataset-size",
        type=int,
        default=10000,
        help="Number of vectors to use from dataset (default: 10000)",
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=None,
        help="Number of IVF clusters (default: auto-calculated as dataset_size/100)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of nearest neighbors to return (default: 5)",
    )
    parser.add_argument(
        "--nprobe",
        type=int,
        default=10,
        help="Number of clusters to search (default: 10)",
    )

    args = parser.parse_args()

    test = FullE2ETest(
        dataset_size=args.dataset_size,
        n_clusters=args.n_clusters,
        k=args.k,
        nprobe=args.nprobe,
    )
    test.run()


if __name__ == "__main__":
    main()
