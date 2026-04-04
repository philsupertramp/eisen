#![cfg(feature = "bf16")]
use eisen::tensor::{Device, Storage};
use eisen::graph::Graph;
use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg};
use std::sync::Arc;

fn setup_gpu() -> Option<Device> {
    match CudaContext::new(0) {
        Ok(ctx) => Some(Device::Gpu(ctx.clone(), ctx.default_stream())),
        Err(_) => None,
    }
}

#[test]
fn test_bf16_kernel_roundtrip() {
    let device = match setup_gpu() {
        Some(d) => d,
        None => {
            eprintln!("No GPU found, skipping BF16 test.");
            return;
        }
    };

    let mut g = Graph::new(device.clone());
    
    // 1. Allocate standard FP32 tensor with explicit fractional values
    let input_data = vec![1.5, -2.25, 3.125, 0.0];
    let a_id = g.alloc(vec![4], input_data.clone());
    
    let (ctx, stream) = match &device {
        Device::Gpu(c, s) => (c, s),
        _ => unreachable!(),
    };

    // 2. Allocate raw BF16 storage
    let bf16_storage = stream.alloc_zeros::<u16>(4).unwrap();
    
    // 3. Dispatch `cast_f32_to_bf16`
    let f_cast = g.functions.get("cast_f32_to_bf16").expect("BF16 cast kernel missing. Check build.rs sm_89 flag.");
    let a_s = match &g.tensors[a_id].data {
        Storage::Gpu(s) => s,
        _ => unreachable!(),
    };

    let n = 4u64;
    let mut builder = stream.launch_builder(f_cast);
    builder.arg(a_s).arg(&bf16_storage).arg(&n);
    unsafe { builder.launch(LaunchConfig::for_num_elems(4)) }.unwrap();

    // 4. Dispatch `cast_bf16_to_f32` to reconstruct the data
    let f32_reconstructed = stream.alloc_zeros::<f32>(4).unwrap();
    let f_reconstruct = g.functions.get("cast_bf16_to_f32").expect("BF16 reconstruct kernel missing.");
    
    let mut builder2 = stream.launch_builder(f_reconstruct);
    builder2.arg(&bf16_storage).arg(&f32_reconstructed).arg(&n);
    unsafe { builder2.launch(LaunchConfig::for_num_elems(4)) }.unwrap();

    // 5. Verify data integrity (Values should be exact for these specific power-of-2 fractions even in 16-bit)
    let reconstructed_data = stream.clone_dtoh(&f32_reconstructed).unwrap();
    for (original, reconstructed) in input_data.iter().zip(reconstructed_data.iter()) {
        assert!((original - reconstructed).abs() < 1e-4, "BF16 precision loss exceeded threshold or kernel failed.");
    }
}

#[test]
fn test_storage_enum_regression() {
    let device = match setup_gpu() {
        Some(d) => d,
        None => return,
    };
    
    let mut g = Graph::new(device);

    // Verify that the addition of GpuBf16 to the Storage enum hasn't disrupted 
    // the standard FP32 memory mapping or Autograd dispatch.
    let a_id = g.alloc(vec![2], vec![1.0, 2.0]);
    let b_id = g.alloc(vec![2], vec![3.0, 4.0]);
    
    let c_id = g.add(a_id, b_id);
    let c_data = g.tensors[c_id].sync_to_cpu();
    
    assert_eq!(c_data, vec![4.0, 6.0]);

    g.backward(c_id);
    let a_grad = g.sync_grad_to_cpu(a_id);
    
    assert_eq!(a_grad, vec![1.0, 1.0]);
}
