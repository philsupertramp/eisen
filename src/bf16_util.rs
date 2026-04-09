// src/bf16_util.rs
//
// Utilities for the BF16 mixed-precision path.
//
// The key invariant across the whole engine:
//   - *Parameter* weights:      Storage::Gpu (FP32) or Storage::GpuBf16 (BF16)
//   - *Activation* data:        Storage::Gpu (FP32) or Storage::GpuBf16 (BF16)
//                               (BF16 iff the Graph is in Bf16Mixed mode)
//   - *Gradient* buffers:       Storage::Gpu (FP32) always
//                               (parameter grads must be FP32 for AdamW stability)
//   - *Optimizer moment* bufs:  CudaSlice<u16> (BF16) in Bf16Mixed mode
//                               (already handled by adamw_step_bf16mom_f32 kernel)
//
// This module exposes one free function used heavily in backward closures:
//
//   bf16_to_f32_temp(storage, size, stream, cast_fn)
//     -> Option<CudaSlice<f32>>
//
//   Returns Some(temp_f32) if `storage` is GpuBf16 (caller must keep alive
//   for the duration of the kernel that reads it, then drop).
//   Returns None if `storage` is already Gpu<f32> (use the slice directly).
//
// Usage pattern in a backward closure:
//
//   let x_tmp = bf16_to_f32_temp(&tensors[x_id].data, size, &stream, &f_cast);
//   let x_f32: &CudaSlice<f32> = match (&x_tmp, &tensors[x_id].data) {
//       (Some(t), _)             => t,
//       (None, Storage::Gpu(s))  => s,
//       _                        => unreachable!(),
//   };
//   // ... launch kernel with x_f32 ...
//   drop(x_tmp); // frees the temp VRAM

use std::sync::Arc;
use cudarc::driver::{CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use crate::tensor::Storage;

/// Cast a BF16 activation to a fresh FP32 VRAM buffer.
/// Returns `None` if the storage is already FP32 (no-op path).
///
/// The returned slice is ephemeral — it lives only for the backward kernel
/// and must be dropped immediately after (to keep peak VRAM low).
#[cfg(feature = "bf16")]
pub fn bf16_to_f32_temp(
    storage: &Storage,
    size: usize,
    stream: &Arc<CudaStream>,
    cast_fn: &CudaFunction,
) -> Option<CudaSlice<f32>> {
    match storage {
        Storage::GpuBf16(s) => {
            let temp = stream
                .alloc_zeros::<f32>(size)
                .expect("bf16_to_f32_temp: alloc failed");
            let n = size as u64;
            let mut b = stream.launch_builder(cast_fn);
            b.arg(s).arg(&temp).arg(&n);
            unsafe { b.launch(LaunchConfig::for_num_elems(size as u32)) }
                .expect("bf16_to_f32_temp: cast kernel failed");
            Some(temp)
        }
        _ => None,
    }
}

/// Cast a BF16 activation to a fresh FP32 VRAM buffer (feature-gated stub).
/// Without bf16 feature this always returns None; the caller falls through to
/// the FP32 arm which is always true in that case.
#[cfg(not(feature = "bf16"))]
pub fn bf16_to_f32_temp(
    _storage: &Storage,
    _size: usize,
    _stream: &Arc<CudaStream>,
    _cast_fn: &CudaFunction,
) -> Option<CudaSlice<f32>> {
    None
}

/// Helper: does this storage tag indicate BF16?
pub fn is_bf16(s: &Storage) -> bool {
    #[cfg(feature = "bf16")]
    {
        matches!(s, Storage::GpuBf16(_))
    }
    #[cfg(not(feature = "bf16"))]
    {
        false
    }
}
