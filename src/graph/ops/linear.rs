use crate::graph::{Graph, TapeNode, is_bf16};
use crate::safe_bf16_temp;
use crate::tensor::{Tensor, Device, Storage};
use cudarc::driver::{PushKernelArg, LaunchConfig, CudaFunction};
use std::collections::HashMap;

fn matmul_kernels(
    functions: &HashMap<String, CudaFunction>, 
    a: &Storage, 
    b: &Storage
) -> (CudaFunction, CudaFunction, CudaFunction) 
{
    #[cfg(feature = "bf16")]
    match (is_bf16(a), is_bf16(b)) {
        (true, true) => (
            functions["matmul_bf16_f32"].clone(),
            functions["matmul_backward_a_bf16b_f32"].clone(),
            functions["matmul_backward_b_f32"].clone(),
        ),
        _ => (
            functions["matmul_f32"].clone(),
            functions["matmul_backward_a_f32"].clone(),
            functions["matmul_backward_b_f32"].clone(),
        ),
    }
    #[cfg(not(feature = "bf16"))]
    (
        functions["matmul_f32"].clone(),
        functions["matmul_backward_a_f32"].clone(),
        functions["matmul_backward_b_f32"].clone(),
    )
}

impl Graph {
    pub fn matmul(&mut self, a_id: usize, b_id: usize) -> usize {
        // Streaming dispatch: if b is CPU-homed (weight streaming), use the
        // streaming path which htods b on demand and frees it immediately.
        let b_is_cpu = matches!(&self.tensors[b_id].data, Storage::Cpu(_));
        #[cfg(feature = "bf16")]
        let a_is_gpu = matches!(&self.tensors[a_id].data, Storage::Gpu(_) | Storage::GpuBf16(_));
        #[cfg(not(feature = "bf16"))]
        let a_is_gpu = matches!(&self.tensors[a_id].data, Storage::Gpu(_));

        if a_is_gpu && b_is_cpu {
            return self.matmul_streamed(a_id, b_id);
        }

        let a_shape = self.tensors[a_id].shape.clone();
        let b_shape = self.tensors[b_id].shape.clone();
        assert!(a_shape.len() >= 2, "matmul: lhs must have rank >= 2");
        assert_eq!(b_shape.len(), 2, "matmul: rhs must have rank 2 [k, n]");
        let k = *a_shape.last().unwrap();
        let m = a_shape[..a_shape.len() - 1].iter().product::<usize>();
        let n = b_shape[1];
        assert_eq!(
            b_shape[0], k,
            "matmul: lhs last dim must equal rhs first dim"
        );
        let device = self.device.clone();

        match &device {
            Device::Gpu(_, stream) => {
                let (f_fwd, f_bwd_a, f_bwd_b) = matmul_kernels(
                    &self.functions,
                    &self.tensors[a_id].data,
                    &self.tensors[b_id].data,
                );
                let stream_clone = stream.clone();

                let out_id = self.alloc_pooled(vec![m, n]);

                #[cfg(feature = "bf16")]
                let (compute_target_id, cast_to_bf16_after) = if self.uses_bf16_mixed_precision() {
                    // allocate a ephemeral FP32 compute buffer (not pooled, owned by this scope)
                    let stream = match &self.device { Device::Gpu(_, s) => s.clone(), _ => unreachable!() };
                    let f32_slice = self.safe_alloc_zeros::<f32>(&stream, m * n);
                    let tmp_id = self.tensors.len();
                    let grad_slice = self.safe_alloc_zeros::<f32>(&stream, 1);
                    self.tensors.push(Tensor {
                        id: tmp_id, shape: vec![m, n],
                        strides: Tensor::compute_strides(&[m, n]),
                        data: Storage::Gpu(f32_slice),
                        grad: Storage::Gpu(grad_slice), // unused
                        device: self.device.clone(), name: None, is_pooled: false,
                    });
                    (tmp_id, true)
                } else {
                    (out_id, false)
                };
                #[cfg(not(feature = "bf16"))]
                #[allow(unused_variables)]
                let (compute_target_id, _cast_to_bf16_after) = (out_id, false);

                #[cfg(feature = "bf16")]
                let f_cast_to_f32 = self.functions.get("cast_bf16_to_f32").unwrap().clone();
                #[cfg(feature = "bf16")]
                let f_cast_to_f32_bwd = f_cast_to_f32.clone();

                #[cfg(feature = "bf16")]
                let a_temp_fwd = safe_bf16_temp!(self, a_id, m * k, &stream, &f_cast_to_f32);
                #[cfg(not(feature = "bf16"))]
                let a_temp_fwd: Option<()> = None;

                #[cfg(feature = "bf16")]
                let b_temp_fwd = safe_bf16_temp!(self, b_id, k * n, &stream, &f_cast_to_f32);
                #[cfg(not(feature = "bf16"))]
                let b_temp_fwd: Option<()> = None;

                self.ensure_on_gpu(a_id);
                self.ensure_on_gpu(b_id);
                self.ensure_on_gpu(compute_target_id);

                let a_s = match (&self.tensors[a_id].data, &a_temp_fwd) {
                    (Storage::Gpu(s), _) => s,
                    #[cfg(feature = "bf16")] (_, Some(t)) => t,
                    _ => unreachable!(),
                };
                let b_s = match (&self.tensors[b_id].data, &b_temp_fwd) {
                    (Storage::Gpu(s), _) => s,
                    #[cfg(feature = "bf16")] (_, Some(t)) => t,
                    _ => unreachable!(),
                };
                let o_s = match &self.tensors[compute_target_id].data {
                    Storage::Gpu(s) => s,
                    _ => unreachable!(),
                };

                let m_u64 = m as u64;
                let k_u64 = k as u64;
                let n_u64 = n as u64;

                let mut builder = stream.launch_builder(&f_fwd);
                builder
                    .arg(a_s)
                    .arg(b_s)
                    .arg(o_s)
                    .arg(&m_u64)
                    .arg(&k_u64)
                    .arg(&n_u64);

                let cfg = LaunchConfig {
                    grid_dim: ((n as u32 + 15) / 16, (m as u32 + 15) / 16, 1),
                    block_dim: (16, 16, 1),
                    shared_mem_bytes: 0,
                };
                unsafe { builder.launch(cfg) }.unwrap_or_else(|err| {
                    panic!("matmul forward kernel launch failed: {:?} (m={}, k={}, n={}, grid={:?}, block={:?})", err, m, k, n, cfg.grid_dim, cfg.block_dim)
                });

                let backward_fn = Box::new(move |tensors: &mut [Tensor]| {
                    let out_grad = match &tensors[out_id].grad {
                        Storage::Gpu(s) => s,
                        _ => unreachable!(),
                    };
                    let a_grad = match &tensors[a_id].grad {
                        Storage::Gpu(s) => s,
                        _ => unreachable!(),
                    };
                    let b_grad = match &tensors[b_id].grad {
                        Storage::Gpu(s) => s,
                        _ => unreachable!(),
                    };
                    
                    #[cfg(feature = "bf16")]
                    let a_temp_bwd = crate::bf16_util::bf16_to_f32_temp(&tensors[a_id].data, m * k, &stream_clone, &f_cast_to_f32_bwd);
                    #[cfg(not(feature = "bf16"))]
                    let a_temp_bwd: Option<()> = None;

                    #[cfg(feature = "bf16")]
                    let b_temp_bwd = crate::bf16_util::bf16_to_f32_temp(&tensors[b_id].data, k * n, &stream_clone, &f_cast_to_f32_bwd);
                    #[cfg(not(feature = "bf16"))]
                    let b_temp_bwd: Option<()> = None;

                    let a_data = match (&tensors[a_id].data, &a_temp_bwd) {
                        (Storage::Gpu(s), _) => s,
                        #[cfg(feature = "bf16")] (_, Some(t)) => t,
                        _ => unreachable!(),
                    };
                    let b_data = match (&tensors[b_id].data, &b_temp_bwd) {
                        (Storage::Gpu(s), _) => s,
                        #[cfg(feature = "bf16")] (_, Some(t)) => t,
                        _ => unreachable!(),
                    };

                    let cfg_a = LaunchConfig {
                        grid_dim: ((k as u32 + 15) / 16, (m as u32 + 15) / 16, 1),
                        block_dim: (16, 16, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut b1 = stream_clone.launch_builder(&f_bwd_a);
                    b1.arg(out_grad)
                        .arg(b_data)
                        .arg(a_grad)
                        .arg(&m_u64)
                        .arg(&k_u64)
                        .arg(&n_u64);
                    unsafe { b1.launch(cfg_a) }.unwrap_or_else(|err| {
                        panic!("matmul backward a kernel launch failed: {:?} (m={}, k={}, n={}, grid={:?}, block={:?})", err, m, k, n, cfg_a.grid_dim, cfg_a.block_dim)
                    });

                    let cfg_b = LaunchConfig {
                        grid_dim: ((n as u32 + 15) / 16, (k as u32 + 15) / 16, 1),
                        block_dim: (16, 16, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut b2 = stream_clone.launch_builder(&f_bwd_b);
                    b2.arg(a_data)
                        .arg(out_grad)
                        .arg(b_grad)
                        .arg(&m_u64)
                        .arg(&k_u64)
                        .arg(&n_u64);
                    unsafe { b2.launch(cfg_b) }.unwrap_or_else(|err| {
                        panic!("matmul backward b kernel launch failed: {:?} (m={}, k={}, n={}, grid={:?}, block={:?})", err, m, k, n, cfg_b.grid_dim, cfg_b.block_dim)
                    });

                    // Sync to ensure kernels finish reading local temp buffers before they drop!
                    stream_clone.synchronize().unwrap_or_else(|err| {
                        panic!("matmul backward sync failed: {:?}", err)
                    });
                });
                
                #[cfg(feature = "bf16")]
                if cast_to_bf16_after {
                    let f_cast = self.functions.get("cast_f32_to_bf16").unwrap().clone();
                    let stream = match &self.device { Device::Gpu(_, s) => s.clone(), _ => unreachable!() };
                    let n_elem = (m * n) as u64;
                    match (&self.tensors[compute_target_id].data, &self.tensors[out_id].data) {
                        (Storage::Gpu(f32_src), Storage::GpuBf16(bf16_dst)) => {
                            let mut b = stream.launch_builder(&f_cast);
                            b.arg(f32_src).arg(bf16_dst).arg(&n_elem);
                            let cfg_cast = LaunchConfig::for_num_elems((m * n) as u32);
                            unsafe { b.launch(cfg_cast) }.unwrap_or_else(|err| {
                                panic!("matmul cast_f32_to_bf16 kernel launch failed: {:?} (m={}, n={}, grid={:?}, block={:?})", err, m, n, cfg_cast.grid_dim, cfg_cast.block_dim)
                            });
                            stream.synchronize().unwrap_or_else(|err| {
                                panic!("matmul post-cast synchronize failed: {:?}", err)
                            });
                        }
                        _ => {}
                    }
                    // Remove the ephemeral FP32 buffer (it's at the end of tensors)
                    self.tensors.pop(); // drops the CudaSlice → cudaFree
                }
                if !self.no_grad {
                    self.tape.nodes.push(TapeNode {
                        inputs: vec![a_id, b_id],
                        output: out_id,
                        backward_fn,
                    });
                }
                out_id
            }
            Device::Cpu => {
                let mut out_data = vec![0.0; m * n];
                let a_fwd = self.tensors[a_id].data.as_cpu().clone();
                let b_fwd = self.tensors[b_id].data.as_cpu().clone();
                for r in 0..m {
                    for c in 0..n {
                        let mut sum = 0.0;
                        for i in 0..k {
                            sum += a_fwd[r * k + i] * b_fwd[i * n + c];
                        }
                        out_data[r * n + c] = sum;
                    }
                }
                let out_id = self.alloc(vec![m, n], out_data);
                let backward_fn = Box::new(move |tensors: &mut [Tensor]| {
                    let out_grad = tensors[out_id].grad.as_cpu().clone();
                    let a_grad = tensors[a_id].grad.as_cpu_mut();
                    for r in 0..m {
                        for i in 0..k {
                            let mut sum = 0.0;
                            for c in 0..n {
                                sum += out_grad[r * n + c] * b_fwd[i * n + c];
                            }
                            a_grad[r * k + i] += sum;
                        }
                    }
                    let b_grad = tensors[b_id].grad.as_cpu_mut();
                    for i in 0..k {
                        for c in 0..n {
                            let mut sum = 0.0;
                            for r in 0..m {
                                sum += a_fwd[r * k + i] * out_grad[r * n + c];
                            }
                            b_grad[i * n + c] += sum;
                        }
                    }
                });
                if !self.no_grad {
                    self.tape.nodes.push(TapeNode {
                        inputs: vec![a_id, b_id],
                        output: out_id,
                        backward_fn,
                    });
                }
                out_id
            }
        }
    }

    pub fn matmul_trans_b(&mut self, a_id: usize, b_id: usize) -> usize {
        // 1. Streaming dispatch (consistent with matmul)
        let b_is_cpu = matches!(&self.tensors[b_id].data, Storage::Cpu(_));
        #[cfg(feature = "bf16")]
        let a_is_gpu = matches!(&self.tensors[a_id].data, Storage::Gpu(_) | Storage::GpuBf16(_));
        #[cfg(not(feature = "bf16"))]
        let a_is_gpu = matches!(&self.tensors[a_id].data, Storage::Gpu(_));

        if a_is_gpu && b_is_cpu {
            return self.matmul_trans_b_streamed(a_id, b_id);
        }

        let a_shape = self.tensors[a_id].shape.clone();
        let b_shape = self.tensors[b_id].shape.clone();
        
        assert!(a_shape.len() >= 2, "matmul_trans_b: lhs must have rank >= 2");
        assert_eq!(b_shape.len(), 2, "matmul_trans_b: rhs must have rank 2 [n, k]");
        
        let k = *a_shape.last().unwrap();
        let m = a_shape[..a_shape.len() - 1].iter().product::<usize>();
        let n = b_shape[0]; 
        
        assert_eq!(
            b_shape[1], k,
            "matmul_trans_b: lhs last dim must equal rhs last dim (k)"
        );

        let device = self.device.clone();

        match &device {
            Device::Gpu(_, stream) => {
                let stream_clone = stream.clone();

                // Resolve kernels (sticking to the naming convention)
                let f_fwd = self.functions.get("matmul_trans_b_f32").expect("matmul_trans_b_f32 kernel not found").clone();
                let f_bwd_a = self.functions.get("matmul_f32").expect("matmul_f32 kernel not found").clone();
                let f_bwd_b = self.functions.get("matmul_trans_a_f32").expect("matmul_trans_a_f32 kernel not found").clone();

                #[cfg(feature = "bf16")]
                let f_cast_to_f32 = self.functions.get("cast_bf16_to_f32").unwrap().clone();
                #[cfg(feature = "bf16")]
                let f_cast_to_f32_bwd = f_cast_to_f32.clone();


                let mut out_shape = a_shape.clone();
                *out_shape.last_mut().unwrap() = n;
                let out_id = self.alloc_pooled(out_shape);

                #[cfg(feature = "bf16")]
                let (compute_target_id, cast_to_bf16_after) = if self.uses_bf16_mixed_precision() {
                    let stream = match &self.device { Device::Gpu(_, s) => s.clone(), _ => unreachable!() };
                    let f32_slice = self.safe_alloc_zeros::<f32>(&stream, m * n);
                    let tmp_id = self.tensors.len();
                    let grad_slice = self.safe_alloc_zeros::<f32>(&stream, 1);
                    self.tensors.push(Tensor {
                        id: tmp_id, shape: vec![m, n],
                        strides: Tensor::compute_strides(&[m, n]),
                        data: Storage::Gpu(f32_slice),
                        grad: Storage::Gpu(grad_slice),
                        device: self.device.clone(), name: None, is_pooled: false,
                    });
                    (tmp_id, true)
                } else {
                    (out_id, false)
                };
                #[cfg(not(feature = "bf16"))]
                #[allow(unused_variables)]
                let (compute_target_id, _cast_to_bf16_after) = (out_id, false);

                self.ensure_on_gpu(a_id);
                self.ensure_on_gpu(b_id);
                self.ensure_on_gpu(compute_target_id);

                let m_u64 = m as u64;
                let k_u64 = k as u64;
                let n_u64 = n as u64;

                // --- FORWARD ---
                {
                    #[cfg(feature = "bf16")]
                    let a_temp_fwd = safe_bf16_temp!(self, a_id, m * k, &stream, &f_cast_to_f32);
                    #[cfg(not(feature = "bf16"))]
                    let a_temp_fwd: Option<()> = None;

                    #[cfg(feature = "bf16")]
                    let b_temp_fwd = safe_bf16_temp!(self, b_id, k * n, &stream, &f_cast_to_f32);
                    #[cfg(not(feature = "bf16"))]
                    let b_temp_fwd: Option<()> = None;

                    let a_s = match (&self.tensors[a_id].data, &a_temp_fwd) {
                        (Storage::Gpu(s), _) => s,
                        #[cfg(feature = "bf16")] (_, Some(t)) => t,
                        _ => unreachable!(),
                    };
                    let b_s = match (&self.tensors[b_id].data, &b_temp_fwd) {
                        (Storage::Gpu(s), _) => s,
                        #[cfg(feature = "bf16")] (_, Some(t)) => t,
                        _ => unreachable!(),
                    };
                    let o_s = match &self.tensors[compute_target_id].data {
                        Storage::Gpu(s) => s,
                        _ => unreachable!("matmul_trans_b: compute target must be GPU FP32"),
                    };

                    let mut builder = stream.launch_builder(&f_fwd);
                    builder.arg(a_s).arg(b_s).arg(o_s).arg(&m_u64).arg(&k_u64).arg(&n_u64);

                    let cfg = LaunchConfig {
                        grid_dim: ((n as u32 + 15) / 16, (m as u32 + 15) / 16, 1),
                        block_dim: (16, 16, 1),
                        shared_mem_bytes: 0,
                    };
                    unsafe { builder.launch(cfg) }.unwrap_or_else(|err| {
                        panic!("matmul_trans_b forward kernel launch failed: {:?} (m={}, k={}, n={}, grid={:?}, block={:?})", err, m, k, n, cfg.grid_dim, cfg.block_dim)
                    });
                }

                // --- BACKWARD ---
                if !self.no_grad {
                    let backward_fn = Box::new(move |tensors: &mut [Tensor]| {
                        let out_grad = match &tensors[out_id].grad {
                            Storage::Gpu(s) => s,
                            _ => unreachable!(),
                        };
                        let a_grad = match &tensors[a_id].grad {
                            Storage::Gpu(s) => s,
                            _ => unreachable!(),
                        };
                        let b_grad = match &tensors[b_id].grad {
                            Storage::Gpu(s) => s,
                            _ => unreachable!(),
                        };
                        
                        #[cfg(feature = "bf16")]
                        let a_temp_bwd = crate::bf16_util::bf16_to_f32_temp(&tensors[a_id].data, m * k, &stream_clone, &f_cast_to_f32_bwd);
                        #[cfg(not(feature = "bf16"))]
                        let a_temp_bwd: Option<()> = None;

                        #[cfg(feature = "bf16")]
                        let b_temp_bwd = crate::bf16_util::bf16_to_f32_temp(&tensors[b_id].data, k * n, &stream_clone, &f_cast_to_f32_bwd);
                        #[cfg(not(feature = "bf16"))]
                        let b_temp_bwd: Option<()> = None;

                        let a_data = match (&tensors[a_id].data, &a_temp_bwd) {
                            (Storage::Gpu(s), _) => s,
                            #[cfg(feature = "bf16")] (_, Some(t)) => t,
                            _ => unreachable!(),
                        };
                        let b_data = match (&tensors[b_id].data, &b_temp_bwd) {
                            (Storage::Gpu(s), _) => s,
                            #[cfg(feature = "bf16")] (_, Some(t)) => t,
                            _ => unreachable!(),
                        };

                        // 1. dA = dC * B  (Standard Matmul)
                        let mut b1 = stream_clone.launch_builder(&f_bwd_a);
                        b1.arg(out_grad).arg(b_data).arg(a_grad).arg(&m_u64).arg(&n_u64).arg(&k_u64);
                        let cfg_a = LaunchConfig {
                            grid_dim: ((k as u32 + 15) / 16, (m as u32 + 15) / 16, 1),
                            block_dim: (16, 16, 1),
                            shared_mem_bytes: 0,
                        };

                        unsafe { b1.launch(cfg_a) }.unwrap_or_else(|err| {
                            panic!("matmul_trans_b backward a kernel launch failed: {:?} (m={}, k={}, n={}, grid={:?}, block={:?})", err, m, k, n, cfg_a.grid_dim, cfg_a.block_dim)
                        });
                        // Sync to ensure kernels finish reading local temp buffers before they drop!
                        stream_clone.synchronize().unwrap_or_else(|err| {
                            panic!("matmul_trans_b backward sync failed: {:?}", err)
                        });

                        // 2. dB = dC^T * A (Transposed A Matmul)
                        let mut b2 = stream_clone.launch_builder(&f_bwd_b);
                        b2.arg(out_grad).arg(a_data).arg(b_grad).arg(&m_u64).arg(&n_u64).arg(&k_u64);
                        let cfg_b = LaunchConfig {
                            grid_dim: ((k as u32 + 15) / 16, (n as u32 + 15) / 16, 1),
                            block_dim: (16, 16, 1),
                            shared_mem_bytes: 0,
                        };
                        unsafe { b2.launch(cfg_b) }.unwrap_or_else(|err| {
                            panic!("matmul_trans_b backward b kernel launch failed: {:?} (m={}, k={}, n={}, grid={:?}, block={:?})", err, m, k, n, cfg_b.grid_dim, cfg_b.block_dim)
                        });

                        // Sync to ensure kernels finish reading local temp buffers before they drop!
                        stream_clone.synchronize().unwrap_or_else(|err| {
                            panic!("matmul_trans_b backward sync failed: {:?}", err)
                        });
                    });

                    self.tape.nodes.push(TapeNode {
                        inputs: vec![a_id, b_id],
                        output: out_id,
                        backward_fn,
                    });
                }

                #[cfg(feature = "bf16")]
                if cast_to_bf16_after {
                    let f_cast = self.functions.get("cast_f32_to_bf16").unwrap().clone();
                    let stream = match &self.device { Device::Gpu(_, s) => s.clone(), _ => unreachable!() };
                    let n_elem = (m * n) as u64;
                    match (&self.tensors[compute_target_id].data, &self.tensors[out_id].data) {
                        (Storage::Gpu(f32_src), Storage::GpuBf16(bf16_dst)) => {
                            let mut b = stream.launch_builder(&f_cast);
                            b.arg(f32_src).arg(bf16_dst).arg(&n_elem);
                            let cfg_cast = LaunchConfig::for_num_elems((m * n) as u32);
                            unsafe { b.launch(cfg_cast) }.unwrap_or_else(|err| {
                                panic!("matmul_trans_b cast_f32_to_bf16 kernel launch failed: {:?} (m={}, n={}, grid={:?}, block={:?})", err, m, n, cfg_cast.grid_dim, cfg_cast.block_dim)
                            });
                            stream.synchronize().unwrap_or_else(|err| {
                                panic!("matmul_trans_b post-cast synchronize failed: {:?}", err)
                            });
                        }
                        _ => {}
                    }
                    self.tensors.pop();
                }

                out_id
            }
            Device::Cpu => {
                // ... (existing CPU logic would go here)
                unimplemented!("CPU matmul_trans_b not implemented yet")
            }
        }
    }

    /// Mixed-precision matrix multiplication.
    ///
    /// Forward:  A_fp32 → BF16  |  B_fp32 → BF16  |  BF16 × BF16 → FP32 out
    /// Backward: standard FP32 matmul gradients (no loss scaling required for BF16)
    ///
    /// Master weights (B) always stay FP32, so the AdamW optimizer and the
    /// rest of the graph are completely unmodified.
    ///
    /// BF16 quantization happens on-the-fly in the forward kernel, so
    /// no full-size BF16 staging tensors are allocated in VRAM.
    #[cfg(feature = "bf16")]
    pub fn matmul_bf16(&mut self, a_id: usize, b_id: usize) -> usize {
        // Streaming dispatch: if b is CPU-homed (weight streaming), use the
        // streaming path which htods b on demand and frees it immediately.
        let b_is_cpu = matches!(&self.tensors[b_id].data, Storage::Cpu(_));
        let a_is_gpu = matches!(&self.tensors[a_id].data, Storage::Gpu(_) | Storage::GpuBf16(_));

        if a_is_gpu && b_is_cpu {
            return self.matmul_streamed(a_id, b_id);
        }
        let a_shape = self.tensors[a_id].shape.clone();
        let b_shape = self.tensors[b_id].shape.clone();
        assert!(a_shape.len() >= 2, "matmul_bf16: lhs must have rank >= 2");
        assert_eq!(b_shape.len(), 2, "matmul_bf16: rhs must have rank 2 [k, n]");
        let k = *a_shape.last().unwrap();
        let m = a_shape[..a_shape.len() - 1].iter().product::<usize>();
        let n = b_shape[1];
        assert_eq!(
            b_shape[0], k,
            "matmul_bf16: lhs last dim must equal rhs first dim"
        );
        let device = self.device.clone();
        self.ensure_on_gpu(b_id);

        match &device {
            Device::Gpu(_, stream) => {
                let f_matmul_fp32rhs = self
                    .functions
                    .get("matmul_f32_bf16accum_f32")
                    .unwrap()
                    .clone();
                let f_matmul_bf16rhs = self
                    .functions
                    .get("matmul_f32_bf16rhsaccum_f32")
                    .unwrap()
                    .clone();
                // Backward uses the existing tiled FP32 kernels — no new kernel needed.
                let f_bwd_a_f32 = self.functions.get("matmul_backward_a_f32").unwrap().clone();
                let f_bwd_a_bf16 = self
                    .functions
                    .get("matmul_backward_a_bf16b_f32")
                    .unwrap()
                    .clone();
                let f_bwd_b = self.functions.get("matmul_backward_b_f32").unwrap().clone();
                let stream_clone = stream.clone();

                // Allocate output first to avoid aliasing immutable tensor borrows
                // with a mutable borrow of `self`.
                let out_id = self.alloc_pooled(vec![m, n]);

                #[cfg(feature = "bf16")]
                let (compute_target_id, cast_to_bf16_after) = if self.uses_bf16_mixed_precision() {
                    let stream = match &self.device { Device::Gpu(_, s) => s.clone(), _ => unreachable!() };
                    let f32_slice = self.safe_alloc_zeros::<f32>(&stream, m * n);
                    let tmp_id = self.tensors.len();
                    let grad_slice = self.safe_alloc_zeros::<f32>(&stream, 1);
                    self.tensors.push(Tensor {
                        id: tmp_id, shape: vec![m, n],
                        strides: Tensor::compute_strides(&[m, n]),
                        data: Storage::Gpu(f32_slice),
                        grad: Storage::Gpu(grad_slice),
                        device: self.device.clone(), name: None, is_pooled: false,
                    });
                    (tmp_id, true)
                } else {
                    (out_id, false)
                };
                #[cfg(not(feature = "bf16"))]
                #[allow(unused_variables)]
                let (compute_target_id, _cast_to_bf16_after) = (out_id, false);

                #[cfg(feature = "bf16")]
                let f_cast_to_f32 = self.functions.get("cast_bf16_to_f32").unwrap().clone();
                #[cfg(feature = "bf16")]
                let f_cast_to_f32_bwd = f_cast_to_f32.clone();

                #[cfg(feature = "bf16")]
                let a_temp_fwd = safe_bf16_temp!(self, a_id, m * k, stream, &f_cast_to_f32);
                #[cfg(not(feature = "bf16"))]
                let a_temp_fwd: Option<()> = None;

                let a_fp32 = match (&self.tensors[a_id].data, &a_temp_fwd) {
                    (Storage::Gpu(s), _) => s,
                    #[cfg(feature = "bf16")] (_, Some(t)) => t,
                    _ => unreachable!("matmul_bf16: a_id must be Gpu or GpuBf16"),
                };

                // ── BF16-style compute (on-the-fly quantization) → FP32 output ─────
                let o_fp32 = match &self.tensors[compute_target_id].data {
                    Storage::Gpu(s) => s,
                    _ => unreachable!(),
                };
                let m_u64 = m as u64;
                let k_u64 = k as u64;
                let n_u64 = n as u64;
                let cfg_fwd = LaunchConfig {
                    grid_dim: ((n as u32 + 15) / 16, (m as u32 + 15) / 16, 1),
                    block_dim: (16, 16, 1),
                    shared_mem_bytes: 0,
                };

                match &self.tensors[b_id].data {
                    Storage::Gpu(b_fp32) => {
                        let mut builder = stream.launch_builder(&f_matmul_fp32rhs);
                        builder
                            .arg(a_fp32)
                            .arg(b_fp32)
                            .arg(o_fp32)
                            .arg(&m_u64)
                            .arg(&k_u64)
                            .arg(&n_u64);
                        unsafe { builder.launch(cfg_fwd) }.unwrap_or_else(|err| {
                            panic!("matmul_bf16 forward kernel launch failed (fp32 rhs): {:?} (m={}, k={}, n={}, grid={:?}, block={:?})", err, m, k, n, cfg_fwd.grid_dim, cfg_fwd.block_dim)
                        });
                    }
                    Storage::GpuBf16(b_bf16) => {
                        let mut builder = stream.launch_builder(&f_matmul_bf16rhs);
                        builder
                            .arg(a_fp32)
                            .arg(b_bf16)
                            .arg(o_fp32)
                            .arg(&m_u64)
                            .arg(&k_u64)
                            .arg(&n_u64);
                        unsafe { builder.launch(cfg_fwd) }.unwrap_or_else(|err| {
                            panic!("matmul_bf16 forward kernel launch failed (bf16 rhs): {:?} (m={}, k={}, n={}, grid={:?}, block={:?})", err, m, k, n, cfg_fwd.grid_dim, cfg_fwd.block_dim)
                        });
                    }
                    _ => unreachable!(),
                }

                // ── Backward closure ─────────────────────────────────────────────────
                //
                // Gradients are computed entirely in FP32 using the existing tiled
                // kernels (matmul_backward_a_f32 / matmul_backward_b_f32).
                let backward_fn = Box::new(move |tensors: &mut [Tensor]| {
                    let out_grad = match &tensors[out_id].grad {
                        Storage::Gpu(s) => s,
                        _ => unreachable!(),
                    };
                    let a_grad = match &tensors[a_id].grad {
                        Storage::Gpu(s) => s,
                        _ => unreachable!(),
                    };
                    let b_grad = match &tensors[b_id].grad {
                        Storage::Gpu(s) => s,
                        _ => unreachable!(),
                    };
                    
                    #[cfg(feature = "bf16")]
                    let a_temp_bwd = crate::bf16_util::bf16_to_f32_temp(&tensors[a_id].data, m * k, &stream_clone, &f_cast_to_f32_bwd);
                    #[cfg(not(feature = "bf16"))]
                    let a_temp_bwd: Option<()> = None;

                    let a_data = match (&tensors[a_id].data, &a_temp_bwd) {
                        (Storage::Gpu(s), _) => s,
                        #[cfg(feature = "bf16")] (_, Some(t)) => t,
                        _ => unreachable!("matmul_bf16 backward: a_id must be Gpu or GpuBf16"),
                    };

                    // grad_a = grad_out @ B^T
                    let cfg_a = LaunchConfig {
                        grid_dim: ((k as u32 + 15) / 16, (m as u32 + 15) / 16, 1),
                        block_dim: (16, 16, 1),
                        shared_mem_bytes: 0,
                    };
                    match &tensors[b_id].data {
                        Storage::Gpu(b_data) => {
                            let mut b1 = stream_clone.launch_builder(&f_bwd_a_f32);
                            b1.arg(out_grad)
                                .arg(b_data)
                                .arg(a_grad)
                                .arg(&m_u64)
                                .arg(&k_u64)
                                .arg(&n_u64);
                            unsafe { b1.launch(cfg_a) }.unwrap_or_else(|err| {
                                panic!("matmul_bf16 backward a kernel launch failed (fp32): {:?} (m={}, k={}, n={}, grid={:?}, block={:?})", err, m, k, n, cfg_a.grid_dim, cfg_a.block_dim)
                            });
                        }
                        Storage::GpuBf16(b_data) => {
                            let mut b1 = stream_clone.launch_builder(&f_bwd_a_bf16);
                            b1.arg(out_grad)
                                .arg(b_data)
                                .arg(a_grad)
                                .arg(&m_u64)
                                .arg(&k_u64)
                                .arg(&n_u64);
                            unsafe { b1.launch(cfg_a) }.unwrap_or_else(|err| {
                                panic!("matmul_bf16 backward a kernel launch failed (bf16): {:?} (m={}, k={}, n={}, grid={:?}, block={:?})", err, m, k, n, cfg_a.grid_dim, cfg_a.block_dim)
                            });
                        }
                        _ => unreachable!(),
                    }

                    // grad_b = A^T @ grad_out
                    let cfg_b = LaunchConfig {
                        grid_dim: ((n as u32 + 15) / 16, (k as u32 + 15) / 16, 1),
                        block_dim: (16, 16, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut b2 = stream_clone.launch_builder(&f_bwd_b);
                    b2.arg(a_data)
                        .arg(out_grad)
                        .arg(b_grad)
                        .arg(&m_u64)
                        .arg(&k_u64)
                        .arg(&n_u64);
                    unsafe { b2.launch(cfg_b) }.unwrap_or_else(|err| {
                        panic!("matmul_bf16 backward b kernel launch failed: {:?} (m={}, k={}, n={}, grid={:?}, block={:?})", err, m, k, n, cfg_b.grid_dim, cfg_b.block_dim)
                    });

                    // Sync to ensure kernels finish reading local temp buffers before they drop!
                    stream_clone.synchronize().unwrap_or_else(|err| {
                        panic!("matmul_bf16 backward sync failed: {:?}", err)
                    });
                });

                #[cfg(feature = "bf16")]
                if cast_to_bf16_after {
                    let f_cast = self.functions.get("cast_f32_to_bf16").unwrap().clone();
                    let stream = match &self.device { Device::Gpu(_, s) => s.clone(), _ => unreachable!() };
                    let n_elem = (m * n) as u64;
                    match (&self.tensors[compute_target_id].data, &self.tensors[out_id].data) {
                        (Storage::Gpu(f32_src), Storage::GpuBf16(bf16_dst)) => {
                            let mut b = stream.launch_builder(&f_cast);
                            b.arg(f32_src).arg(bf16_dst).arg(&n_elem);
                            let cfg_cast = LaunchConfig::for_num_elems((m * n) as u32);
                            unsafe { b.launch(cfg_cast) }.unwrap_or_else(|err| {
                                panic!("matmul_bf16 cast_f32_to_bf16 kernel launch failed: {:?} (m={}, n={}, grid={:?}, block={:?})", err, m, n, cfg_cast.grid_dim, cfg_cast.block_dim)
                            });
                            stream.synchronize().unwrap_or_else(|err| {
                                panic!("matmul_bf16 post-cast synchronize failed: {:?}", err)
                            });
                        }
                        _ => {}
                    }
                    self.tensors.pop();
                }

                if !self.no_grad {
                    self.tape.nodes.push(TapeNode {
                        inputs: vec![a_id, b_id],
                        output: out_id,
                        backward_fn,
                    });
                }
                out_id
            }

            // CPU fallback: BF16 is a GPU-only optimisation.
            // On CPU we silently use the standard FP32 path so tests still pass.
            Device::Cpu => self.matmul(a_id, b_id),
        }
    }

    fn matmul_streamed(&mut self, a_id: usize, b_id: usize) -> usize {
        let a_shape = self.tensors[a_id].shape.clone();
        let b_shape = self.tensors[b_id].shape.clone();
        assert!(
            a_shape.len() >= 2,
            "matmul_streamed: lhs must have rank >= 2"
        );
        assert_eq!(
            b_shape.len(),
            2,
            "matmul_streamed: rhs must have rank 2 [k, n]"
        );
        let k = *a_shape.last().unwrap();
        let m = a_shape[..a_shape.len() - 1].iter().product::<usize>();
        let n = b_shape[1];
        assert_eq!(
            b_shape[0], k,
            "matmul_streamed: lhs last dim must equal rhs first dim"
        );

        let (gpu_device, stream) = match &self.device {
            Device::Gpu(d, s) => (d.clone(), s.clone()),
            Device::Cpu => unreachable!("matmul_streamed called on CPU graph"),
        };

        let (f_fwd, f_bwd_a, f_bwd_b) = matmul_kernels(
            &self.functions,
            &self.tensors[a_id].data,
            &self.tensors[b_id].data,
        );
        let stream_bwd = stream.clone();
        let gpu_device_clone = gpu_device.clone();

        // ── Forward: htod b → kernel → SYNC → FREE ─────────────────────────────
        let b_f32_fwd = self.tensors[b_id].data.to_f32_vec();
        let b_temp_fwd = stream.clone_htod(b_f32_fwd.as_slice()).unwrap_or_else(|err| {
            panic!("matmul_streamed: forward htod failed for size {}: {:?}", b_f32_fwd.len(), err)
        });

        let out_id = self.alloc_pooled(vec![m, n]);

        #[cfg(feature = "bf16")]
        let (compute_target_id, cast_to_bf16_after) = if self.uses_bf16_mixed_precision() {
            let f32_slice = self.safe_alloc_zeros::<f32>(&stream, m * n);
            let tmp_id = self.tensors.len();
            let grad_slice = self.safe_alloc_zeros::<f32>(&stream, 1);
            self.tensors.push(Tensor {
                id: tmp_id, shape: vec![m, n],
                strides: Tensor::compute_strides(&[m, n]),
                data: Storage::Gpu(f32_slice),
                grad: Storage::Gpu(grad_slice),
                device: self.device.clone(), name: None, is_pooled: false,
            });
            (tmp_id, true)
        } else {
            (out_id, false)
        };
        #[cfg(not(feature = "bf16"))]
        #[allow(unused_variables)]
        let (compute_target_id, _cast_to_bf16_after) = (out_id, false);

        #[cfg(feature = "bf16")]
        let f_cast_to_f32 = self.functions.get("cast_bf16_to_f32").unwrap().clone();
        #[cfg(feature = "bf16")]
        let f_cast_to_f32_bwd = f_cast_to_f32.clone();

        #[cfg(feature = "bf16")]
        let a_temp_fwd = safe_bf16_temp!(self, a_id, m * k, &stream, &f_cast_to_f32);
        #[cfg(not(feature = "bf16"))]
        let a_temp_fwd: Option<()> = None;

        let a_s = match (&self.tensors[a_id].data, &a_temp_fwd) {
            (Storage::Gpu(s), _) => s,
            #[cfg(feature = "bf16")] (_, Some(t)) => t,
            _ => unreachable!("matmul_streamed: input a must be GPU storage"),
        };
        let o_s = match &self.tensors[compute_target_id].data {
            Storage::Gpu(s) => s,
            _ => unreachable!(),
        };
        let m_u64 = m as u64;
        let k_u64 = k as u64;
        let n_u64 = n as u64;
        
        let cfg_fwd = LaunchConfig {
            grid_dim: ((n as u32 + 15) / 16, (m as u32 + 15) / 16, 1),
            block_dim: (16, 16, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = stream.launch_builder(&f_fwd);
        builder
            .arg(a_s)
            .arg(&b_temp_fwd)
            .arg(o_s)
            .arg(&m_u64)
            .arg(&k_u64)
            .arg(&n_u64);
        unsafe { builder.launch(cfg_fwd) }.unwrap_or_else(|err| {
            panic!("matmul_streamed forward kernel launch failed: {:?} (m={}, k={}, n={}, grid={:?}, block={:?})", err, m, k, n, cfg_fwd.grid_dim, cfg_fwd.block_dim)
        });

        // Sync before free: guarantees the matmul kernel has finished reading
        stream
            .synchronize()
            .unwrap_or_else(|err| panic!("matmul_streamed: forward sync failed: {:?}", err));
        drop(b_temp_fwd); // cudaFree — now safe

        // ── Backward closure ────────────────────────────────────────────────────
        let backward_fn = Box::new(move |tensors: &mut [Tensor]| {
            let b_f32_bwd = tensors[b_id].data.to_f32_vec();
            let b_temp_bwd = stream_bwd.clone_htod(b_f32_bwd.as_slice()).unwrap_or_else(|err| {
                panic!("matmul_streamed: backward htod failed for size {}: {:?}", b_f32_bwd.len(), err)
            });

            let out_grad = match &tensors[out_id].grad {
                Storage::Gpu(s) => s,
                _ => unreachable!(),
            };
            let a_grad = match &tensors[a_id].grad {
                Storage::Gpu(s) => s,
                _ => unreachable!(),
            };
            
            #[cfg(feature = "bf16")]
            let a_temp_bwd = crate::bf16_util::bf16_to_f32_temp(&tensors[a_id].data, m * k, &stream_bwd, &f_cast_to_f32_bwd);
            #[cfg(not(feature = "bf16"))]
            let a_temp_bwd: Option<()> = None;

            let a_data = match (&tensors[a_id].data, &a_temp_bwd) {
                (Storage::Gpu(s), _) => s,
                #[cfg(feature = "bf16")] (_, Some(t)) => t,
                _ => unreachable!("matmul_streamed: a_data must be GPU"),
            };

            // grad_a = grad_out @ b^T   (GPU → GPU, accumulate in-place)
            let cfg_a = LaunchConfig {
                grid_dim: ((k as u32 + 15) / 16, (m as u32 + 15) / 16, 1),
                block_dim: (16, 16, 1),
                shared_mem_bytes: 0,
            };
            let mut b1 = stream_bwd.launch_builder(&f_bwd_a);
            b1.arg(out_grad)
                .arg(&b_temp_bwd)
                .arg(a_grad)
                .arg(&m_u64)
                .arg(&k_u64)
                .arg(&n_u64);
            unsafe { b1.launch(cfg_a) }.unwrap_or_else(|err| {
                panic!("matmul_streamed backward a kernel launch failed: {:?} (m={}, k={}, n={}, grid={:?}, block={:?})", err, m, k, n, cfg_a.grid_dim, cfg_a.block_dim)
            });

            // Safe, immediate closure-bound VRAM allocation
            let mut grad_b_temp = stream_bwd.alloc_zeros::<f32>(k * n).unwrap_or_else(|err| {
                panic!("matmul_streamed grad_b_temp allocation failed for size {}: {:?}", k * n, err)
            });

            // grad_b = a^T @ grad_out  (GPU temp → dtoh → accumulate into CPU grad)
            let cfg_b = LaunchConfig {
                grid_dim: ((n as u32 + 15) / 16, (k as u32 + 15) / 16, 1),
                block_dim: (16, 16, 1),
                shared_mem_bytes: 0,
            };
            let mut b2 = stream_bwd.launch_builder(&f_bwd_b);
            b2.arg(a_data)
                .arg(out_grad)
                .arg(&mut grad_b_temp)
                .arg(&m_u64)
                .arg(&k_u64)
                .arg(&n_u64);
            unsafe { b2.launch(cfg_b) }.unwrap_or_else(|err| {
                panic!("matmul_streamed backward b kernel launch failed: {:?} (m={}, k={}, n={}, grid={:?}, block={:?})", err, m, k, n, cfg_b.grid_dim, cfg_b.block_dim)
            });

            // Sync before dtoh: the backward kernel must be done before we read
            stream_bwd
                .synchronize()
                .unwrap_or_else(|err| panic!("matmul_streamed: backward sync failed: {:?}", err));

            let grad_b_gpu = stream_bwd
                .clone_dtoh(&grad_b_temp)
                .unwrap_or_else(|err| panic!("matmul_streamed: grad dtoh failed: {:?}", err));

            // Free GPU temporaries now that data is on CPU
            drop(b_temp_bwd);
            // Accumulate into the CPU grad buffer 
            let b_grad = tensors[b_id].grad.as_cpu_mut();
            for (acc, delta) in b_grad.iter_mut().zip(grad_b_gpu.iter()) {
                *acc += delta;
            }
        });

        if !self.no_grad {
            self.tape.nodes.push(TapeNode {
                inputs: vec![a_id, b_id],
                output: out_id,
                backward_fn,
            });
        }

        #[cfg(feature = "bf16")]
        if cast_to_bf16_after {
            let f_cast = self.functions.get("cast_f32_to_bf16").unwrap().clone();
            let n_elem = (m * n) as u64;
            match (&self.tensors[compute_target_id].data, &self.tensors[out_id].data) {
                (Storage::Gpu(f32_src), Storage::GpuBf16(bf16_dst)) => {
                    let mut b = stream.launch_builder(&f_cast);
                    b.arg(f32_src).arg(bf16_dst).arg(&n_elem);
                    let cfg_cast = LaunchConfig::for_num_elems((m * n) as u32);
                    unsafe { b.launch(cfg_cast) }.unwrap_or_else(|err| {
                        panic!("matmul_streamed cast_f32_to_bf16 kernel launch failed: {:?} (m={}, n={}, grid={:?}, block={:?})", err, m, n, cfg_cast.grid_dim, cfg_cast.block_dim)
                    });
                    stream.synchronize().unwrap_or_else(|err| {
                        panic!("matmul_streamed post-cast synchronize failed: {:?}", err)
                    });
                }
                _ => {}
            }
            self.tensors.pop(); // Safely pops compute_target_id!
        }

        out_id
    }

    fn matmul_trans_b_streamed(&mut self, a_id: usize, b_id: usize) -> usize {
        let a_shape = self.tensors[a_id].shape.clone();
        let b_shape = self.tensors[b_id].shape.clone();
        
        assert!(a_shape.len() >= 2, "matmul_trans_b_streamed: lhs must have rank >= 2");
        assert_eq!(b_shape.len(), 2, "matmul_trans_b_streamed: rhs must have rank 2 [n, k]");
        
        let k = *a_shape.last().unwrap();
        let m = a_shape[..a_shape.len() - 1].iter().product::<usize>();
        let n = b_shape[0]; 
        
        assert_eq!(
            b_shape[1], k,
            "matmul_trans_b_streamed: lhs last dim must equal rhs last dim (k)"
        );

        let (gpu_device, stream) = match &self.device {
            Device::Gpu(d, s) => (d.clone(), s.clone()),
            Device::Cpu => unreachable!("matmul_trans_b_streamed called on CPU graph"),
        };

        let f_fwd = self.functions.get("matmul_trans_b_f32").expect("matmul_trans_b_f32 kernel not found").clone();
        let f_bwd_a = self.functions.get("matmul_f32").expect("matmul_f32 kernel not found").clone();
        let f_bwd_b = self.functions.get("matmul_trans_a_f32").expect("matmul_trans_a_f32 kernel not found").clone();
        let stream_bwd = stream.clone();
        let gpu_device_clone = gpu_device.clone();

        // ── Forward: htod b → kernel → SYNC → FREE ─────────────────────────────
        let b_f32_fwd = self.tensors[b_id].data.to_f32_vec();
        let b_temp_fwd = stream.clone_htod(b_f32_fwd.as_slice()).unwrap_or_else(|err| {
            panic!("matmul_trans_b_streamed: forward htod failed for size {}: {:?}", b_f32_fwd.len(), err)
        });

        let mut out_shape = a_shape.clone();
        *out_shape.last_mut().unwrap() = n;
        let out_id = self.alloc_pooled(out_shape);

        #[cfg(feature = "bf16")]
        let (compute_target_id, cast_to_bf16_after) = if self.uses_bf16_mixed_precision() {
            let f32_slice = self.safe_alloc_zeros::<f32>(&stream, m * n);
            let tmp_id = self.tensors.len();
            let grad_slice = self.safe_alloc_zeros::<f32>(&stream, 1);
            self.tensors.push(Tensor {
                id: tmp_id, shape: vec![m, n],
                strides: Tensor::compute_strides(&[m, n]),
                data: Storage::Gpu(f32_slice),
                grad: Storage::Gpu(grad_slice),
                device: self.device.clone(), name: None, is_pooled: false,
            });
            (tmp_id, true)
        } else {
            (out_id, false)
        };
        #[cfg(not(feature = "bf16"))]
        #[allow(unused_variables)]
        let (compute_target_id, _cast_to_bf16_after) = (out_id, false);

        #[cfg(feature = "bf16")]
        let f_cast_to_f32 = self.functions.get("cast_bf16_to_f32").unwrap().clone();
        #[cfg(feature = "bf16")]
        let f_cast_to_f32_bwd = f_cast_to_f32.clone();

        #[cfg(feature = "bf16")]
        let a_temp_fwd = safe_bf16_temp!(self, a_id, m * k, &stream, &f_cast_to_f32);
        #[cfg(not(feature = "bf16"))]
        let a_temp_fwd: Option<()> = None;

        let a_s = match (&self.tensors[a_id].data, &a_temp_fwd) {
            (Storage::Gpu(s), _) => s,
            #[cfg(feature = "bf16")] (_, Some(t)) => t,
            _ => unreachable!("matmul_trans_b_streamed: input a must be GPU storage"),
        };
        let o_s = match &self.tensors[compute_target_id].data {
            Storage::Gpu(s) => s,
            _ => unreachable!(),
        };
        let m_u64 = m as u64;
        let k_u64 = k as u64;
        let n_u64 = n as u64;
        
        let cfg_fwd = LaunchConfig {
            grid_dim: ((n as u32 + 15) / 16, (m as u32 + 15) / 16, 1),
            block_dim: (16, 16, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = stream.launch_builder(&f_fwd);
        builder
            .arg(a_s)
            .arg(&b_temp_fwd)
            .arg(o_s)
            .arg(&m_u64)
            .arg(&k_u64)
            .arg(&n_u64);
        unsafe { builder.launch(cfg_fwd) }.unwrap_or_else(|err| {
            panic!("matmul_trans_b_streamed forward kernel launch failed: {:?} (m={}, k={}, n={}, grid={:?}, block={:?})", err, m, k, n, cfg_fwd.grid_dim, cfg_fwd.block_dim)
        });

        // Sync before free: guarantees the matmul kernel has finished reading
        stream
            .synchronize()
            .unwrap_or_else(|err| panic!("matmul_trans_b_streamed: forward sync failed: {:?}", err));
        drop(b_temp_fwd); // cudaFree — now safe

        // ── Backward closure ────────────────────────────────────────────────────
        let backward_fn = Box::new(move |tensors: &mut [Tensor]| {
            let b_f32_bwd = tensors[b_id].data.to_f32_vec();
            let b_temp_bwd = stream_bwd.clone_htod(b_f32_bwd.as_slice()).unwrap_or_else(|err| {
                panic!("matmul_trans_b_streamed: backward htod failed for size {}: {:?}", b_f32_bwd.len(), err)
            });

            let out_grad = match &tensors[out_id].grad {
                Storage::Gpu(s) => s,
                _ => unreachable!(),
            };
            let a_grad = match &tensors[a_id].grad {
                Storage::Gpu(s) => s,
                _ => unreachable!(),
            };
            
            #[cfg(feature = "bf16")]
            let a_temp_bwd = crate::bf16_util::bf16_to_f32_temp(&tensors[a_id].data, m * k, &stream_bwd, &f_cast_to_f32_bwd);
            #[cfg(not(feature = "bf16"))]
            let a_temp_bwd: Option<()> = None;

            let a_data = match (&tensors[a_id].data, &a_temp_bwd) {
                (Storage::Gpu(s), _) => s,
                #[cfg(feature = "bf16")] (_, Some(t)) => t,
                _ => unreachable!("matmul_trans_b_streamed: a_data must be GPU"),
            };

            // grad_a = grad_out @ b^T   (GPU → GPU, accumulate in-place)
            let cfg_a = LaunchConfig {
                grid_dim: ((k as u32 + 15) / 16, (m as u32 + 15) / 16, 1),
                block_dim: (16, 16, 1),
                shared_mem_bytes: 0,
            };
            let mut b1 = stream_bwd.launch_builder(&f_bwd_a);
            b1.arg(out_grad)
                .arg(&b_temp_bwd)
                .arg(a_grad)
                .arg(&m_u64)
                .arg(&n_u64)
                .arg(&k_u64);
            unsafe { b1.launch(cfg_a) }.unwrap_or_else(|err| {
                panic!("matmul_trans_b_streamed backward a kernel launch failed: {:?} (m={}, k={}, n={}, grid={:?}, block={:?})", err, m, k, n, cfg_a.grid_dim, cfg_a.block_dim)
            });

            // Safe, immediate closure-bound VRAM allocation
            let mut grad_b_temp = stream_bwd.alloc_zeros::<f32>(n * k).unwrap_or_else(|err| {
                panic!("matmul_trans_b_streamed grad_b_temp allocation failed for size {}: {:?}", n * k, err)
            });

            // grad_b = a^T @ grad_out  (GPU temp → dtoh → accumulate into CPU grad)
            let cfg_b = LaunchConfig {
                grid_dim: ((k as u32 + 15) / 16, (n as u32 + 15) / 16, 1),
                block_dim: (16, 16, 1),
                shared_mem_bytes: 0,
            };
            let mut b2 = stream_bwd.launch_builder(&f_bwd_b);
            b2.arg(out_grad)
                .arg(a_data)
                .arg(&mut grad_b_temp)
                .arg(&m_u64)
                .arg(&n_u64)
                .arg(&k_u64);
            unsafe { b2.launch(cfg_b) }.unwrap_or_else(|err| {
                panic!("matmul_trans_b_streamed backward b kernel launch failed: {:?} (m={}, k={}, n={}, grid={:?}, block={:?})", err, m, k, n, cfg_b.grid_dim, cfg_b.block_dim)
            });

            stream_bwd.synchronize().unwrap_or_else(|err| panic!("matmul_trans_b_streamed: backward sync failed: {:?}", err));

            let grad_b_gpu = stream_bwd
                .clone_dtoh(&grad_b_temp)
                .unwrap_or_else(|err| panic!("matmul_trans_b_streamed: grad dtoh failed: {:?}", err));

            drop(b_temp_bwd);
            let b_grad = tensors[b_id].grad.as_cpu_mut();
            for (acc, delta) in b_grad.iter_mut().zip(grad_b_gpu.iter()) {
                *acc += delta;
            }
        });

        if !self.no_grad {
            self.tape.nodes.push(TapeNode {
                inputs: vec![a_id, b_id],
                output: out_id,
                backward_fn,
            });
        }

        #[cfg(feature = "bf16")]
        if cast_to_bf16_after {
            let f_cast = self.functions.get("cast_f32_to_bf16").unwrap().clone();
            let n_elem = (m * n) as u64;
            match (&self.tensors[compute_target_id].data, &self.tensors[out_id].data) {
                (Storage::Gpu(f32_src), Storage::GpuBf16(bf16_dst)) => {
                    let mut b = stream.launch_builder(&f_cast);
                    b.arg(f32_src).arg(bf16_dst).arg(&n_elem);
                    let cfg_cast = LaunchConfig::for_num_elems((m * n) as u32);
                    unsafe { b.launch(cfg_cast) }.unwrap_or_else(|err| {
                        panic!("matmul_trans_b_streamed cast_f32_to_bf16 kernel launch failed: {:?} (m={}, n={}, grid={:?}, block={:?})", err, m, n, cfg_cast.grid_dim, cfg_cast.block_dim)
                    });
                    stream.synchronize().unwrap_or_else(|err| {
                        panic!("matmul_trans_b_streamed post-cast synchronize failed: {:?}", err)
                    });
                }
                _ => {}
            }
            self.tensors.pop(); // Safely pops compute_target_id!
        }

        out_id
    }
    
    pub fn bmm(&mut self, a_id: usize, b_id: usize, trans_b: bool) -> usize {
        let a_shape = self.tensors[a_id].shape.clone();
        let b_shape = self.tensors[b_id].shape.clone();
        let batch = a_shape[0];
        let m = a_shape[1];
        let k = a_shape[2];
        let n = if trans_b { b_shape[1] } else { b_shape[2] };
        let device = self.device.clone();

        match &device {
            Device::Gpu(_, stream) => {
                let f_fwd = if self.uses_bf16_mixed_precision() {
                    self.functions.get("bmm_f32_bf16accum_f32").unwrap().clone()
                } else {
                    self.functions.get("bmm_f32").unwrap().clone()
                };
                let f_bwd_a = self
                    .functions
                    .get(if trans_b {
                        "bmm_backward_a_transb_f32"
                    } else {
                        "bmm_backward_a_f32"
                    })
                    .unwrap()
                    .clone();
                let f_bwd_b = self
                    .functions
                    .get(if trans_b {
                        "bmm_backward_b_transb_f32"
                    } else {
                        "bmm_backward_b_f32"
                    })
                    .unwrap()
                    .clone();
                let stream_clone = stream.clone();
                let out_id = self.alloc_pooled(vec![batch, m, n]);

                #[cfg(feature = "bf16")]
                let (compute_target_id, cast_to_bf16_after) = if self.uses_bf16_mixed_precision() {
                    // allocate a ephemeral FP32 compute buffer (not pooled, owned by this scope)
                    let stream = match &self.device { Device::Gpu(_, s) => s.clone(), _ => unreachable!() };
                    let f32_slice = self.safe_alloc_zeros::<f32>(&stream, batch * m * n);
                    let tmp_id = self.tensors.len();
                    let grad_slice = self.safe_alloc_zeros::<f32>(&stream, 1);
                    self.tensors.push(Tensor {
                        id: tmp_id, shape: vec![batch, m, n],
                        strides: Tensor::compute_strides(&[batch, m, n]),
                        data: Storage::Gpu(f32_slice),
                        grad: Storage::Gpu(grad_slice),
                        device: self.device.clone(), name: None, is_pooled: false,
                    });
                    (tmp_id, true)
                } else {
                    (out_id, false)
                };
                #[cfg(not(feature = "bf16"))]
                #[allow(unused_variables)]
                let (compute_target_id, _cast_to_bf16_after) = (out_id, false);

                #[cfg(feature = "bf16")]
                let f_cast_to_f32 = self.functions.get("cast_bf16_to_f32").unwrap().clone();
                #[cfg(feature = "bf16")]
                let f_cast_to_f32_bwd = f_cast_to_f32.clone();

                #[cfg(feature = "bf16")]
                let a_temp_fwd = safe_bf16_temp!(self, a_id, batch * m * k, &stream, &f_cast_to_f32);
                #[cfg(not(feature = "bf16"))]
                let a_temp_fwd: Option<()> = None;

                #[cfg(feature = "bf16")]
                let b_temp_fwd = safe_bf16_temp!(self, b_id, batch * k * n, &stream, &f_cast_to_f32);
                #[cfg(not(feature = "bf16"))]
                let b_temp_fwd: Option<()> = None;

                self.ensure_on_gpu(a_id);
                self.ensure_on_gpu(b_id);
                self.ensure_on_gpu(compute_target_id);

                let a_s = match (&self.tensors[a_id].data, &a_temp_fwd) {
                    (Storage::Gpu(s), _) => s,
                    #[cfg(feature = "bf16")] (_, Some(t)) => t,
                    _ => unreachable!(),
                };
                let b_s = match (&self.tensors[b_id].data, &b_temp_fwd) {
                    (Storage::Gpu(s), _) => s,
                    #[cfg(feature = "bf16")] (_, Some(t)) => t,
                    _ => unreachable!(),
                };
                let o_s = match &self.tensors[compute_target_id].data {
                    Storage::Gpu(s) => s,
                    _ => unreachable!(),
                };

                let batch_u64 = batch as u64;
                let m_u64 = m as u64;
                let k_u64 = k as u64;
                let n_u64 = n as u64;

                let mut builder = stream.launch_builder(&f_fwd);
                builder
                    .arg(a_s)
                    .arg(b_s)
                    .arg(o_s)
                    .arg(&batch_u64)
                    .arg(&m_u64)
                    .arg(&k_u64)
                    .arg(&n_u64)
                    .arg(&trans_b);
                let cfg_fwd = LaunchConfig {
                    grid_dim: ((n as u32 + 15) / 16, (m as u32 + 15) / 16, batch as u32),
                    block_dim: (16, 16, 1),
                    shared_mem_bytes: 0,
                };
                unsafe { builder.launch(cfg_fwd) }.unwrap_or_else(|err| {
                    panic!("bmm forward kernel launch failed: {:?} (batch={}, m={}, k={}, n={}, trans_b={}, grid={:?}, block={:?})", err, batch, m, k, n, trans_b, cfg_fwd.grid_dim, cfg_fwd.block_dim)
                });

                let backward_fn = Box::new(move |tensors: &mut [Tensor]| {
                    let out_grad = match &tensors[out_id].grad {
                        Storage::Gpu(s) => s,
                        _ => unreachable!(),
                    };
                    let a_grad = match &tensors[a_id].grad {
                        Storage::Gpu(s) => s,
                        _ => unreachable!(),
                    };
                    let b_grad = match &tensors[b_id].grad {
                        Storage::Gpu(s) => s,
                        _ => unreachable!(),
                    };
                    
                    #[cfg(feature = "bf16")]
                    let a_temp_bwd = crate::bf16_util::bf16_to_f32_temp(&tensors[a_id].data, batch * m * k, &stream_clone, &f_cast_to_f32_bwd);
                    #[cfg(not(feature = "bf16"))]
                    let a_temp_bwd: Option<()> = None;

                    #[cfg(feature = "bf16")]
                    let b_temp_bwd = crate::bf16_util::bf16_to_f32_temp(&tensors[b_id].data, batch * k * n, &stream_clone, &f_cast_to_f32_bwd);
                    #[cfg(not(feature = "bf16"))]
                    let b_temp_bwd: Option<()> = None;

                    let a_data = match (&tensors[a_id].data, &a_temp_bwd) {
                        (Storage::Gpu(s), _) => s,
                        #[cfg(feature = "bf16")] (_, Some(t)) => t,
                        _ => unreachable!(),
                    };
                    let b_data = match (&tensors[b_id].data, &b_temp_bwd) {
                        (Storage::Gpu(s), _) => s,
                        #[cfg(feature = "bf16")] (_, Some(t)) => t,
                        _ => unreachable!(),
                    };

                    let mut b1 = stream_clone.launch_builder(&f_bwd_a);
                    b1.arg(out_grad)
                        .arg(b_data)
                        .arg(a_grad)
                        .arg(&batch_u64)
                        .arg(&m_u64)
                        .arg(&k_u64)
                        .arg(&n_u64);
                    let cfg_a = LaunchConfig {
                        grid_dim: ((k as u32 + 15) / 16, (m as u32 + 15) / 16, batch as u32),
                        block_dim: (16, 16, 1),
                        shared_mem_bytes: 0,
                    };
                    unsafe { b1.launch(cfg_a) }.unwrap_or_else(|err| {
                        panic!("bmm backward a kernel launch failed: {:?} (batch={}, m={}, k={}, n={}, trans_b={}, grid={:?}, block={:?})", err, batch, m, k, n, trans_b, cfg_a.grid_dim, cfg_a.block_dim)
                    });

                    let mut b2 = stream_clone.launch_builder(&f_bwd_b);
                    b2.arg(a_data)
                        .arg(out_grad)
                        .arg(b_grad)
                        .arg(&batch_u64)
                        .arg(&m_u64)
                        .arg(&k_u64)
                        .arg(&n_u64);
                    let cfg_b = if trans_b {
                        LaunchConfig {
                            grid_dim: ((k as u32 + 15) / 16, (n as u32 + 15) / 16, batch as u32),
                            block_dim: (16, 16, 1),
                            shared_mem_bytes: 0,
                        }
                    } else {
                        LaunchConfig {
                            grid_dim: ((n as u32 + 15) / 16, (k as u32 + 15) / 16, batch as u32),
                            block_dim: (16, 16, 1),
                            shared_mem_bytes: 0,
                        }
                    };
                    unsafe { b2.launch(cfg_b) }.unwrap_or_else(|err| {
                        panic!("bmm backward b kernel launch failed: {:?} (batch={}, m={}, k={}, n={}, trans_b={}, grid={:?}, block={:?})", err, batch, m, k, n, trans_b, cfg_b.grid_dim, cfg_b.block_dim)
                    });

                    // Sync to ensure kernels finish reading local temp buffers before they drop!
                    stream_clone.synchronize().unwrap_or_else(|err| {
                        panic!("bmm backward sync failed: {:?}", err)
                    });
                });
                #[cfg(feature = "bf16")]
                if cast_to_bf16_after {
                    let f_cast = self.functions.get("cast_f32_to_bf16").unwrap().clone();
                    let stream = match &self.device { Device::Gpu(_, s) => s.clone(), _ => unreachable!() };
                    let n_elem = (batch * m * n) as u64;
                    match (&self.tensors[compute_target_id].data, &self.tensors[out_id].data) {
                        (Storage::Gpu(f32_src), Storage::GpuBf16(bf16_dst)) => {
                            let mut b = stream.launch_builder(&f_cast);
                            b.arg(f32_src).arg(bf16_dst).arg(&n_elem);
                            let cfg_cast = LaunchConfig::for_num_elems((batch * m * n) as u32);
                            unsafe { b.launch(cfg_cast) }.unwrap_or_else(|err| {
                                panic!("bmm cast_f32_to_bf16 kernel launch failed: {:?} (batch={}, m={}, n={}, grid={:?}, block={:?})", err, batch, m, n, cfg_cast.grid_dim, cfg_cast.block_dim)
                            });
                            stream.synchronize().unwrap_or_else(|err| {
                                panic!("bmm post-cast synchronize failed: {:?}", err)
                            });
                        }
                        _ => {}
                    }
                    // Remove the ephemeral FP32 buffer (it's at the end of tensors)
                    self.tensors.pop(); // drops the CudaSlice → cudaFree
                }
                if !self.no_grad {
                    self.tape.nodes.push(TapeNode {
                        inputs: vec![a_id, b_id],
                        output: out_id,
                        backward_fn,
                    });
                }
                out_id
            }
            Device::Cpu => {
                let a_data = self.tensors[a_id].data.as_cpu().clone();
                let b_data = self.tensors[b_id].data.as_cpu().clone();
                let mut out_data = vec![0.0; batch * m * n];

                for bb in 0..batch {
                    for i in 0..m {
                        for j in 0..n {
                            let mut acc = 0.0;
                            for kk in 0..k {
                                let a_idx = ((bb * m + i) * k) + kk;
                                let b_idx = if trans_b {
                                    ((bb * n + j) * k) + kk
                                } else {
                                    ((bb * k + kk) * n) + j
                                };
                                acc += a_data[a_idx] * b_data[b_idx];
                            }
                            out_data[(bb * m + i) * n + j] = acc;
                        }
                    }
                }

                let out_id = self.alloc(vec![batch, m, n], out_data);
                let backward_fn = Box::new(move |tensors: &mut [Tensor]| {
                    let out_grad = tensors[out_id].grad.as_cpu().clone();

                    if a_id == b_id {
                        let grad = tensors[a_id].grad.as_cpu_mut();
                        for bb in 0..batch {
                            for i in 0..m {
                                for j in 0..n {
                                    let go = out_grad[(bb * m + i) * n + j];
                                    for kk in 0..k {
                                        let a_idx = ((bb * m + i) * k) + kk;
                                        let b_idx = if trans_b {
                                            ((bb * n + j) * k) + kk
                                        } else {
                                            ((bb * k + kk) * n) + j
                                        };
                                        grad[a_idx] += go * b_data[b_idx];
                                        grad[b_idx] += go * a_data[a_idx];
                                    }
                                }
                            }
                        }
                    } else {
                        let (a_grad, b_grad) = if a_id < b_id {
                            let (left, right) = tensors.split_at_mut(b_id);
                            (left[a_id].grad.as_cpu_mut(), right[0].grad.as_cpu_mut())
                        } else {
                            let (left, right) = tensors.split_at_mut(a_id);
                            (right[0].grad.as_cpu_mut(), left[b_id].grad.as_cpu_mut())
                        };

                        for bb in 0..batch {
                            for i in 0..m {
                                for j in 0..n {
                                    let go = out_grad[(bb * m + i) * n + j];
                                    for kk in 0..k {
                                        let a_idx = ((bb * m + i) * k) + kk;
                                        let b_idx = if trans_b {
                                            ((bb * n + j) * k) + kk
                                        } else {
                                            ((bb * k + kk) * n) + j
                                        };
                                        a_grad[a_idx] += go * b_data[b_idx];
                                        b_grad[b_idx] += go * a_data[a_idx];
                                    }
                                }
                            }
                        }
                    }
                });

                if !self.no_grad {
                    self.tape.nodes.push(TapeNode {
                        inputs: vec![a_id, b_id],
                        output: out_id,
                        backward_fn,
                    });
                }
                out_id
            }
        }
    }
}
