from faiss_s3 import generate_meta_from_file


def main():
    file_path = "../exp/faiss-s3-rs/example.ivf"
    meta = generate_meta_from_file(file_path)
    print(meta)
    # assert meta.cluster_data_offset == 3154059


if __name__ == "__main__":
    main()
