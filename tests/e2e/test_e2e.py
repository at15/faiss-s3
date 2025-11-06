# e2e test using the quora question dataset
# - download the dataset
# - create embeddings
# - create index
#   - write to file
#   - generate the meata using faiss_s3's meta.py
#   - upload files to s3
# There should be following files under same path prefix in s3:
# - embeddings.pt # numpy array
# - meta.json # json file with metadata about the index and embeddings
# - attributes.parquet # parquet file saving the source text
# - index.ivf # faiss index file IVF files
#
# - query by downloading the full index.ivf (similar to test_quora.py)
#   - download meta.json so we know what embedding model we were using
#   - download attributes.parquet so we can map the matched vector id back to the source text
#   - no need to download embeddings.pt unless we want to generate a new index
#
# - query by using the faiss_s3 client (similar to test_s3_cache_server.py)
#   - make sure the server is running
#   - download meta.json so we know embedding model and cluster data offset
#   - download attributes.parquet so we can map the matched vector id back to the source text
#   - NO need to download index.ivf, the server will fetch on demand
#
# We should be able to run the tests full e2e or only write/query
# remember to meausre the times spent on each stage and inner steps so it is a simple benchmark
import os


def download_dataset():
    dataset_path = "quora_duplicate_questions.tsv"
    url = "http://qim.fs.quoracdn.net/quora_duplicate_questions.tsv"
    if not os.path.exists(dataset_path):
        from sentence_transformers import util

        print(f"Downloading dataset from {url} to {dataset_path}")
        util.http_get(url, dataset_path)
    return dataset_path


def load_embeddings():
    dataset_path = download_dataset()
    with open(dataset_path, "r") as f:
        for line in f:
            print(line)


def main():
    download_dataset()


if __name__ == "__main__":
    main()
