#[cxx::bridge]
mod ffi {
    extern "Rust" {}

    // Defines what is exposted on C++ side, which are all unsafe
    unsafe extern "C++" {
        include!("faiss_s3.h"); // Used in the generated lib.rs.h

        fn create_example_ivf_index(index_file_name: &str);
    }
}

pub fn create_example_ivf_index(index_file_name: &str) {
    ffi::create_example_ivf_index(index_file_name);
}
