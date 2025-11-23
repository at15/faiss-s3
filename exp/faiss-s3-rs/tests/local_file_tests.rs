#[test]
fn test_ivf_local_file() {
    // faiss_s3_rs::create_example_ivf_index("example.ivf");
    // let offset = faiss_s3_rs::get_cluster_data_offset("example.ivf");
    // 52139, matches the python output from tests/test_meta.py
    // println!("Cluster data offset: {:?}", offset);
    faiss_s3_rs::search_example_ivf_index("example.ivf");
}
