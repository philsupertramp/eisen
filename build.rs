use std::process::Command;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=./kernels/ops.cu");

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    
    // Compile the CUDA kernel to PTX
    // Ensure nvcc is in your PATH
    let mut command = Command::new("nvcc");
    command
        .arg("-ptx")
        .arg("./kernels/ops.cu")
        .arg("-o")
        .arg(out_dir.join("ops.ptx"));
    if std::env::var("CARGO_FEATURE_BF16").is_ok() {
        command.arg("-arch=sm_89");
    }
    let status = command
        .status()
        .expect("Failed to run nvcc. Is the CUDA Toolkit installed and in your PATH?");

    if !status.success() {
        panic!("nvcc failed to compile src/kernels/ops.cu");
    }
}
