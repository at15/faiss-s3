#[cxx::bridge]
mod ffi {
    extern "Rust" {} // TODO: expose things from Rust to C++

    // Defines what is exposted on C++ side, which are all unsafe
    unsafe extern "C++" {
        include!("faiss_s3.h"); // Used in the generated lib.rs.h

        fn CreateExampleIVFIndex(index_file_name: &str);
        fn CreateExampleIVFIndexWithData(
            index_file_name: &str,
            dim: usize,
            n_vectors: usize,
            data: Vec<f32>,
            n_clusters: usize,
        );
        fn SearchExampleIVFIndex(index_file_name: &str);
        fn GetClusterDataOffset(index_file_name: &str) -> Result<usize>;

        type FaissIVFIndexS3;

        fn ClusterSizes(self: &FaissIVFIndexS3) -> Vec<u64>;

        fn CreateFaissIVFIndexS3(
            index_without_cluster_data: Vec<u8>,
        ) -> Result<UniquePtr<FaissIVFIndexS3>>;
    }
}

pub fn create_example_ivf_index(index_file_name: &str) {
    ffi::CreateExampleIVFIndex(index_file_name);
}

pub fn create_example_ivf_index_with_data(index_file_name: &str, dim: usize, n_vectors: usize, data: Vec<f32>, n_clusters: usize) {
    ffi::CreateExampleIVFIndexWithData(index_file_name, dim, n_vectors, data, n_clusters);
}

pub fn search_example_ivf_index(index_file_name: &str) {
    ffi::SearchExampleIVFIndex(index_file_name);
}

pub fn get_cluster_data_offset(index_file_name: &str) -> Result<usize, String> {
    ffi::GetClusterDataOffset(index_file_name).map_err(|e| e.to_string())
}

pub fn create_faiss_ivf_index_s3(
    index_without_cluster_data: Vec<u8>,
) -> Result<cxx::UniquePtr<ffi::FaissIVFIndexS3>, String> {
    ffi::CreateFaissIVFIndexS3(index_without_cluster_data)
        .map_err(|e| e.to_string())
}

#[pyo3::pymodule]
// From https://github.com/PyO3/pyo3
// the mod name must match the lib.name in Cargo.toml
mod faiss_s3_rs {
    use pyo3::prelude::*;

    /// Formats the sum of two numbers as string.
    #[pyfunction]
    fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
        Ok((a + b).to_string())
    }

    /// Create an example IVF index and save to the given file name.
    #[pyfunction]
    fn create_example_ivf_index(index_file_name: &str) -> PyResult<()> {
        crate::create_example_ivf_index(index_file_name);
        Ok(())
    }
}
