use std::path::PathBuf;

fn main() {
    // NOTE: We skip compile because the static library, including the bridge
    // cpp main.rs.cc, is already compiled by CMake.
    let _ = cxx_build::bridge("src/lib.rs");

    // link the C++ library built using CMake
    let cmake_compiled_lib = PathBuf::from("build");
    println!("cargo:rustc-link-search=native={}", cmake_compiled_lib.display());
    println!("cargo:rustc-link-lib=static=faiss-s3-cpp");

    // link faiss
    let faiss_lib = PathBuf::from("deps/faiss-home/lib");
    println!("cargo:rustc-link-search=native={}", faiss_lib.display());
    println!("cargo:rustc-link-lib=static=faiss");

    // link openmp
    // TODO: only works on macOS, linux requires different configuration
    let omp_lib = "/opt/homebrew/opt/libomp/lib";
    println!("cargo:rustc-link-search=native={omp_lib}");
    println!("cargo:rustc-link-lib=dylib=omp"); // links libomp.dylib

    // link BLAS (required by faiss) - use macOS Accelerate framework
    // TODO: only works on macOS
    println!("cargo:rustc-link-lib=framework=Accelerate");

    println!("cargo:rerun-if-changed=src/faiss_s3.cpp");
    println!("cargo:rerun-if-changed=include/faiss_s3.h");
    println!("cargo:rerun-if-changed=CMakeLists.txt");
}
