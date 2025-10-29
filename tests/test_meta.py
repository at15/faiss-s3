from faiss_s3 import generate_meta_from_file


def main():
    meta = generate_meta_from_file(
        "quora-index-quora-distilbert-multilingual-size-100000.idx"
    )
    print(meta)
    assert meta.cluster_data_offset == 3154059


if __name__ == "__main__":
    main()
