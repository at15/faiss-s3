fn main() {
    println!("Hello, world!");
    faiss_s3_rs::create_example_ivf_index("example.ivf");
    let offset = faiss_s3_rs::get_cluster_data_offset("example.ivf");
    // 52139, matches the python output from tests/test_meta.py
    println!("Cluster data offset: {:?}", offset);
}
