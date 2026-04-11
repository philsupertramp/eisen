use std::path::PathBuf;
use std::process::Command;

fn compile_kernel(src: &str, out: &PathBuf, use_bf16: bool) {
    let mut command = Command::new("nvcc");
    command.arg("-ptx").arg(src).arg("-o").arg(out);
    
    if use_bf16 {
        command.arg("-arch=sm_89").arg("-DUSE_BF16_ARITH");
    }
    
    let status = command.status().expect("Failed to run nvcc.");
    if !status.success() {
        panic!("nvcc failed to compile {}", src);
    }
}

fn main() {
    println!("cargo:rerun-if-changed=./kernels/");
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());

    // Always compile standard FP32 kernels
    compile_kernel("./kernels/ops_f32.cu", &out_dir.join("ops_f32.ptx"), false);

    // Conditionally compile BF16 kernels
    if std::env::var("CARGO_FEATURE_BF16").is_ok() {
        compile_kernel("./kernels/ops_bf16.cu", &out_dir.join("ops_bf16.ptx"), true);
    }
}
