use std::process::Command;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=./kernels/ops.cu");

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    
    // Compile the CUDA kernel to PTX
    // Ensure nvcc is in your PATH
    let status = Command::new("nvcc")
        .arg("-ptx")
        .arg("-arch=sm_89") // CRITICAL: Target RTX 40-series (Ada Lovelace) for native BF16 & Tensor Cores
        .arg("./kernels/ops.cu")
        .arg("-o")
        .arg(out_dir.join("ops.ptx"))
        .status()
        .expect("Failed to run nvcc. Is the CUDA Toolkit installed and in your PATH?");

    if !status.success() {
        panic!("nvcc failed to compile src/kernels/ops.cu");
    }
}
