use eisen::graph::Graph;
use eisen::tensor::Device;
use cudarc::driver::{CudaContext};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Eisen Phase 3: GPU Acceleration ===");
    
    // 1. Initialize CUDA
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let device = Device::Gpu(ctx, stream);
    
    // 2. Init Graph on GPU
    let mut g = Graph::new(device);
    
    // 3. Allocate two vectors in VRAM
    let a_id = g.alloc(vec![4], vec![1.0, 2.0, 3.0, 4.0]);
    let b_id = g.alloc(vec![4], vec![10.0, 20.0, 30.0, 40.0]);
    
    // 4. Perform GPU Addition
    let c_id = g.add(a_id, b_id);
    
    // 5. Sync result back to CPU to verify
    let result = g.tensors[c_id].sync_to_cpu();
    println!("GPU Add Result: {:?}", result);
    
    assert_eq!(result, vec![11.0, 22.0, 33.0, 44.0]);
    println!("Verification: SUCCESS");

    Ok(())
}
