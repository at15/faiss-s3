use std::path::PathBuf;

fn main() {
    // NOTE: We skip compile because the static library, including the bridge
    // cpp main.rs.cc, is already compiled by CMake.
    let _ = cxx_build::bridge("src/lib.rs");

    // link the C++ library built using CMake
    let cmake_compiled_lib = PathBuf::from("build");
    println!(
        "cargo:rustc-link-search=native={}",
        cmake_compiled_lib.display()
    );
    println!("cargo:rustc-link-lib=static=faiss-s3-cpp");
}
