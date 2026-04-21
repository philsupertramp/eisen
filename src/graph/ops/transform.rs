use crate::graph::{Graph, TapeNode, is_bf16};
use crate::tensor::{Tensor, Device, Storage};
use cudarc::driver::{PushKernelArg, LaunchConfig};
use crate::safe_bf16_temp;

impl Graph {
    pub fn transpose_0213(&mut self, a_id: usize) -> usize {
        let shape = self.tensors[a_id].shape.clone();
        assert!(shape.len() >= 4, "transpose_0213 requires at least 4 dims");
        let ndim = shape.len();
        let s = shape[ndim - 3];
        let h = shape[ndim - 2];
        let d = shape[ndim - 1];
        let b: usize = shape[..ndim - 3].iter().product();
        let mut out_shape = shape[..ndim - 3].to_vec();
        out_shape.extend_from_slice(&[h, s, d]);
        let size = b * s * h * d;
        let device = self.device.clone();

        match &device {
            Device::Gpu(_, stream) => {
                let stream_clone = stream.clone();

                #[cfg(feature = "bf16")]
                let (f_fwd, f_bwd) = if is_bf16(&self.tensors[a_id].data) {
                    (
                        self.functions.get("transpose_0213_bf16").unwrap().clone(),
                        // backward still FP32: transpose_0213_backward_f32 reads FP32 grad
                        self.functions.get("transpose_0213_backward_f32").unwrap().clone(),
                    )
                } else {
                    (
                        self.functions.get("transpose_0213_f32").unwrap().clone(),
                        self.functions.get("transpose_0213_backward_f32").unwrap().clone(),
                    )
                };
                #[cfg(not(feature = "bf16"))]
                let (f_fwd, f_bwd) = (
                    self.functions.get("transpose_0213_f32").unwrap().clone(),
                    self.functions.get("transpose_0213_backward_f32").unwrap().clone(),
                );

                let out_id = self.alloc_pooled(out_shape);
                let (b_u64, s_u64, h_u64, d_u64) = (b as u64, s as u64, h as u64, d as u64);
                self.ensure_on_gpu(a_id);
                self.ensure_on_gpu(out_id);

                {
                    let mut builder = stream.launch_builder(&f_fwd);
                    match (&self.tensors[a_id].data, &self.tensors[out_id].data) {
                        (Storage::Gpu(a_s), Storage::Gpu(o_s)) => {
                            builder.arg(a_s).arg(o_s).arg(&b_u64).arg(&s_u64).arg(&h_u64).arg(&d_u64);
                        }
                        #[cfg(feature = "bf16")]
                        (Storage::GpuBf16(a_s), Storage::GpuBf16(o_s)) => {
                            builder.arg(a_s).arg(o_s).arg(&b_u64).arg(&s_u64).arg(&h_u64).arg(&d_u64);
                        }
                        _ => panic!("transpose_0213: mismatched storage"),
                    }
                    unsafe { builder.launch(LaunchConfig::for_num_elems(size as u32)) }.unwrap();
                }

                // Backward: FP32 grad_out -> FP32 grad_src (transpose_0213_backward_f32 unchanged)
                let backward_fn = Box::new(move |tensors: &mut [Tensor]| {
                    let out_grad = match &tensors[out_id].grad {
                        Storage::Gpu(s) => s, _ => unreachable!()
                    };
                    let a_grad = match &tensors[a_id].grad {
                        Storage::Gpu(s) => s, _ => unreachable!()
                    };
                    let mut b1 = stream_clone.launch_builder(&f_bwd);
                    b1.arg(out_grad).arg(a_grad)
                        .arg(&b_u64).arg(&s_u64).arg(&h_u64).arg(&d_u64);
                    unsafe { b1.launch(LaunchConfig::for_num_elems(size as u32)) }.unwrap();
                });

                if !self.no_grad {
                    self.tape.nodes.push(TapeNode {
                        inputs: vec![a_id], output: out_id, backward_fn,
                    });
                }
                out_id
            }
            Device::Cpu => {
                let a_data = self.tensors[a_id].data.as_cpu().clone();
                let mut out_data = vec![0.0; size];

                for bb in 0..b {
                    for ss in 0..s {
                        for hh in 0..h {
                            for dd in 0..d {
                                let in_idx = (((bb * s + ss) * h + hh) * d) + dd;
                                let out_idx = (((bb * h + hh) * s + ss) * d) + dd;
                                out_data[out_idx] = a_data[in_idx];
                            }
                        }
                    }
                }

                let out_id = self.alloc(out_shape, out_data);

                let backward_fn = Box::new(move |tensors: &mut [Tensor]| {
                    let out_grad = tensors[out_id].grad.as_cpu().clone();
                    let a_grad = tensors[a_id].grad.as_cpu_mut();

                    for bb in 0..b {
                        for ss in 0..s {
                            for hh in 0..h {
                                for dd in 0..d {
                                    let in_idx = (((bb * s + ss) * h + hh) * d) + dd;
                                    let out_idx = (((bb * h + hh) * s + ss) * d) + dd;
                                    a_grad[in_idx] += out_grad[out_idx];
                                }
                            }
                        }
                    }
                });

                if !self.no_grad {
                    self.tape.nodes.push(TapeNode {
                        inputs: vec![a_id],
                        output: out_id,
                        backward_fn,
                    });
                }

                out_id
            }
        }
    }


    pub fn gather(&mut self, weights_id: usize, indices_id: usize) -> usize {
        let device = self.device.clone();
        match &device {
            Device::Gpu(_, stream) => {
                let w = &self.tensors[weights_id];
                let idx = &self.tensors[indices_id];
                let hidden_dim = w.shape[1];
                let num_indices = idx.data.len();
                let out_size = num_indices * hidden_dim;

                let mut out_shape = idx.shape.clone();
                out_shape.push(hidden_dim);

                let f_fwd_f32 = self.functions.get("gather_f32").unwrap().clone();
                #[allow(unused_variables)]
                let f_bwd = self.functions.get("gather_backward_f32").unwrap().clone();
                let stream_clone = stream.clone();

                let out_id = self.alloc_pooled(out_shape.clone());

                #[cfg(feature = "bf16")]
                let (compute_target_id, cast_to_bf16_after) = if self.uses_bf16_mixed_precision() && !is_bf16(&self.tensors[weights_id].data) {
                    let stream = match &self.device { Device::Gpu(_, s) => s.clone(), _ => unreachable!() };
                    let f32_slice = self.safe_alloc_zeros::<f32>(&stream, out_size);
                    let tmp_id = self.tensors.len();
                    let grad_slice = self.safe_alloc_zeros::<f32>(&stream, 1);
                    self.tensors.push(Tensor {
                        id: tmp_id, shape: out_shape.clone(),
                        strides: Tensor::compute_strides(&out_shape),
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
                let idx_temp_fwd = safe_bf16_temp!(self, indices_id, num_indices, stream, &f_cast_to_f32);
                #[cfg(not(feature = "bf16"))]
                let idx_temp_fwd: Option<()> = None;

                let idx_s = match (&self.tensors[indices_id].data, &idx_temp_fwd) {
                    (Storage::Gpu(s), _) => s,
                    #[cfg(feature = "bf16")] (_, Some(t)) => t,
                    _ => unreachable!("gather: indices must be Gpu or GpuBf16"),
                };
                
                let hidden_u64 = hidden_dim as u64;
                let out_size_u64 = out_size as u64;

                match (
                    &self.tensors[weights_id].data,
                    &self.tensors[compute_target_id].data,
                ) {
                    #[cfg(feature = "bf16")]
                    (Storage::GpuBf16(w_s), Storage::GpuBf16(o_s)) => {
                        let f = self.functions.get("gather_bf16_bf16out").unwrap().clone();
                        let mut builder = stream.launch_builder(&f);
                        builder.arg(w_s).arg(idx_s).arg(o_s)
                            .arg(&hidden_u64).arg(&out_size_u64);
                        unsafe { builder.launch(LaunchConfig::for_num_elems(out_size as u32)) }.unwrap();
                    }
                    (Storage::Gpu(w_s), Storage::Gpu(o_s)) => {
                        let mut builder = stream.launch_builder(&f_fwd_f32);
                        builder
                            .arg(w_s)
                            .arg(idx_s)
                            .arg(o_s)
                            .arg(&hidden_u64)
                            .arg(&out_size_u64);
                        unsafe { builder.launch(LaunchConfig::for_num_elems(out_size as u32)) }
                            .unwrap();
                    }
                    #[cfg(feature = "bf16")]
                    (Storage::GpuBf16(w_s), Storage::Gpu(o_s)) => {
                        let f_fwd_bf16 = self.functions.get("gather_bf16_f32").unwrap().clone();
                        let mut builder = stream.launch_builder(&f_fwd_bf16);
                        builder
                            .arg(w_s)
                            .arg(idx_s)
                            .arg(o_s)
                            .arg(&hidden_u64)
                            .arg(&out_size_u64);
                        unsafe { builder.launch(LaunchConfig::for_num_elems(out_size as u32)) }
                            .unwrap();
                    }
                    _ => panic!("gather: unsupported storage combination"),
                }

                let backward_fn = Box::new(move |tensors: &mut [Tensor]| {
                    #[cfg(feature = "bf16")]
                    let idx_temp_bwd = crate::bf16_util::bf16_to_f32_temp(&tensors[indices_id].data, num_indices, &stream_clone, &f_cast_to_f32_bwd);
                    #[cfg(not(feature = "bf16"))]
                    let idx_temp_bwd: Option<()> = None;

                    let idx_data = match (&tensors[indices_id].data, &idx_temp_bwd) {
                        (Storage::Gpu(s), _) => s,
                        #[cfg(feature = "bf16")] (_, Some(t)) => t,
                        _ => unreachable!("gather: indices must be GPU FP32"),
                    };

                    let out_grad = match &tensors[out_id].grad {
                        Storage::Gpu(s) => s,
                        _ => unreachable!("gather: out_grad must be GPU FP32"),
                    };
                    let w_grad = match &tensors[weights_id].grad {
                        Storage::Gpu(s) => s,
                        _ => unreachable!("gather: w_grad must be GPU FP32"),
                    };

                    let mut b1 = stream_clone.launch_builder(&f_bwd);
                    b1.arg(idx_data)
                        .arg(out_grad)
                        .arg(w_grad)
                        .arg(&hidden_u64)
                        .arg(&out_size_u64);
                    unsafe { b1.launch(LaunchConfig::for_num_elems(out_size as u32)) }.unwrap();
                });

                #[cfg(feature = "bf16")]
                if cast_to_bf16_after {
                    let f_cast = self.functions.get("cast_f32_to_bf16").unwrap().clone();
                    let n_elem = out_size as u64;
                    match (&self.tensors[compute_target_id].data, &self.tensors[out_id].data) {
                        (Storage::Gpu(f32_src), Storage::GpuBf16(bf16_dst)) => {
                            let mut b = stream.launch_builder(&f_cast);
                            b.arg(f32_src).arg(bf16_dst).arg(&n_elem);
                            unsafe { b.launch(LaunchConfig::for_num_elems(out_size as u32)) }.unwrap();
                            stream.synchronize().unwrap();
                        }
                        _ => {}
                    }
                    self.tensors.pop();
                }

                if !self.no_grad {
                    self.tape.nodes.push(TapeNode {
                        inputs: vec![weights_id, indices_id],
                        output: out_id,
                        backward_fn,
                    });
                }
                out_id
            }
            Device::Cpu => {
                let w = &self.tensors[weights_id];
                let idx = &self.tensors[indices_id];
                let hidden_dim = w.shape[1];
                let num_indices = idx.data.as_cpu().len();
                let mut out_data = vec![0.0; num_indices * hidden_dim];
                let idx_data: Vec<usize> = idx.data.as_cpu().iter().map(|&x| x as usize).collect();

                let w_data = w.data.as_cpu();
                for i in 0..num_indices {
                    let row = idx_data[i];
                    for d in 0..hidden_dim {
                        out_data[i * hidden_dim + d] = w_data[row * hidden_dim + d];
                    }
                }

                let mut out_shape = idx.shape.clone();
                out_shape.push(hidden_dim);
                let out_id = self.alloc(out_shape, out_data);
                let backward_fn = Box::new(move |tensors: &mut [Tensor]| {
                    let o_grad = tensors[out_id].grad.as_cpu().clone();
                    let w_grad = tensors[weights_id].grad.as_cpu_mut();
                    for i in 0..num_indices {
                        let row = idx_data[i];
                        for d in 0..hidden_dim {
                            w_grad[row * hidden_dim + d] += o_grad[i * hidden_dim + d];
                        }
                    }
                });
                if !self.no_grad {
                    self.tape.nodes.push(TapeNode {
                        inputs: vec![weights_id, indices_id],
                        output: out_id,
                        backward_fn,
                    });
                }
                out_id
            }
        }
    }


    pub fn reshape(&mut self, a_id: usize, new_shape: Vec<usize>) -> usize {
        let device = self.device.clone();
        match &device {
            Device::Gpu(_, stream) => {
                let old_size = self.tensors[a_id].data.len();
                let stream_clone = stream.clone();

                // Select copy kernel based on storage type
                #[cfg(feature = "bf16")]
                let f_fwd = if is_bf16(&self.tensors[a_id].data) {
                    self.functions.get("copy_bf16").unwrap().clone()
                } else {
                    self.functions.get("copy_f32").unwrap().clone()
                };
                #[cfg(not(feature = "bf16"))]
                let f_fwd = self.functions.get("copy_f32").unwrap().clone();

                let f_bwd = self.functions.get("accumulate_f32").unwrap().clone();

                let out_id = self.alloc_pooled(new_shape);
                let n = old_size as u64;

                self.ensure_on_gpu(a_id);
                self.ensure_on_gpu(out_id);

                {
                    let mut builder = stream.launch_builder(&f_fwd);
                    match (&self.tensors[a_id].data, &self.tensors[out_id].data) {
                        (Storage::Gpu(s), Storage::Gpu(d)) => { builder.arg(s).arg(d).arg(&n); }
                        #[cfg(feature = "bf16")]
                        (Storage::GpuBf16(s), Storage::GpuBf16(d)) => { builder.arg(s).arg(d).arg(&n); }
                        _ => panic!("reshape: mismatched storage types"),
                    }
                    unsafe { builder.launch(LaunchConfig::for_num_elems(old_size as u32)) }.unwrap();
                }

                // Backward: grad is always FP32 — accumulate_f32 unchanged
                let backward_fn = Box::new(move |tensors: &mut [Tensor]| {
                    let out_grad = match &tensors[out_id].grad {
                        Storage::Gpu(s) => s, _ => unreachable!()
                    };
                    let a_grad = match &tensors[a_id].grad {
                        Storage::Gpu(s) => s, _ => unreachable!()
                    };
                    let rank = 1u64; let s0 = n; let s1 = 1u64; let s2 = 1u64;
                    let t0 = 1u64;   let t1 = 1u64; let t2 = 1u64;
                    let mut b1 = stream_clone.launch_builder(&f_bwd);
                    b1.arg(a_grad).arg(out_grad)
                        .arg(&n).arg(&rank)
                        .arg(&s0).arg(&s1).arg(&s2)
                        .arg(&t0).arg(&t1).arg(&t2);
                    unsafe { b1.launch(LaunchConfig::for_num_elems(old_size as u32)) }.unwrap();
                });

                if !self.no_grad {
                    self.tape.nodes.push(TapeNode {
                        inputs: vec![a_id], output: out_id, backward_fn,
                    });
                }
                out_id
            }
            Device::Cpu => {
                let old_size = self.tensors[a_id].data.as_cpu().len();
                let out_id = self.alloc(new_shape, self.tensors[a_id].data.as_cpu().clone());
                let backward_fn = Box::new(move |tensors: &mut [Tensor]| {
                    let o_grad = tensors[out_id].grad.as_cpu().clone();
                    let a_grad = tensors[a_id].grad.as_cpu_mut();
                    for i in 0..old_size {
                        a_grad[i] += o_grad[i];
                    }
                });
                if !self.no_grad {
                    self.tape.nodes.push(TapeNode {
                        inputs: vec![a_id],
                        output: out_id,
                        backward_fn,
                    });
                }
                out_id
            }
        }
    }

    /// Reinterpret tensor metadata as a new contiguous shape without allocating.
    /// Caller must ensure the element count matches.
    pub fn reinterpret_shape(&mut self, tensor_id: usize, new_shape: Vec<usize>) {
        let old_elems = self.tensors[tensor_id].shape.iter().product::<usize>();
        let new_elems = if new_shape.is_empty() {
            1
        } else {
            new_shape.iter().product::<usize>()
        };
        assert_eq!(
            old_elems, new_elems,
            "reinterpret_shape: element count mismatch"
        );
        self.tensors[tensor_id].shape = new_shape.clone();
        self.tensors[tensor_id].strides = Tensor::compute_strides(&new_shape);
    }

    pub fn transpose(&mut self, a_id: usize, dim0: usize, dim1: usize) -> usize {
        match &self.device {
            Device::Gpu(_, _) => {
                panic!("GPU Transpose not natively implemented - use reshape fallback or CPU")
            }
            Device::Cpu => {
                let a = &self.tensors[a_id];
                let mut out_shape = a.shape.clone();
                out_shape.swap(dim0, dim1);
                let mut out_strides = a.strides.clone();
                out_strides.swap(dim0, dim1);
                let mut out_data = vec![0.0; a.data.as_cpu().len()];
                let a_data = a.data.as_cpu();
                for i in 0..out_data.len() {
                    let nd = Tensor::flat_to_nd(i, &out_shape);
                    out_data[i] = a_data[Tensor::nd_to_flat(&nd, &out_strides)];
                }
                let out_id = self.alloc(out_shape, out_data);
                let out_strides_cap = out_strides.clone();
                let backward_fn = Box::new(move |tensors: &mut [Tensor]| {
                    let o_grad = tensors[out_id].grad.as_cpu().clone();
                    for i in 0..o_grad.len() {
                        let nd = Tensor::flat_to_nd(i, &tensors[out_id].shape);
                        tensors[a_id].grad.as_cpu_mut()
                            [Tensor::nd_to_flat(&nd, &out_strides_cap)] += o_grad[i];
                    }
                });
                if !self.no_grad {
                    self.tape.nodes.push(TapeNode {
                        inputs: vec![a_id],
                        output: out_id,
                        backward_fn,
                    });
                }
                out_id
            }
        }
    }

    pub fn rope(&mut self, a_id: usize, head_dim: usize) -> usize {
        let shape     = self.tensors[a_id].shape.clone();
        let seq_len   = shape[1];
        let hidden_dim = shape[2];
        let size      = shape.iter().product::<usize>();
        let num_pairs = size / 2;
        let device    = self.device.clone();

        match &device {
            Device::Gpu(_, stream) => {
                let stream_clone = stream.clone();

                #[cfg(feature = "bf16")]
                let (f_fwd, f_bwd) = if is_bf16(&self.tensors[a_id].data) {
                    (
                        self.functions.get("rope_bf16").unwrap().clone(),
                        self.functions.get("rope_backward_f32").unwrap().clone(),
                    )
                } else {
                    (
                        self.functions.get("rope_f32").unwrap().clone(),
                        self.functions.get("rope_backward_f32").unwrap().clone(),
                    )
                };
                #[cfg(not(feature = "bf16"))]
                let (f_fwd, f_bwd) = (
                    self.functions.get("rope_f32").unwrap().clone(),
                    self.functions.get("rope_backward_f32").unwrap().clone(),
                );

                let out_id = self.alloc_pooled(shape);
                let (s_u64, hd_u64, hdim_u64, np_u64) = (
                    seq_len as u64, hidden_dim as u64, head_dim as u64, num_pairs as u64,
                );
                self.ensure_on_gpu(a_id);
                self.ensure_on_gpu(out_id);

                {
                    let mut builder = stream.launch_builder(&f_fwd);
                    match (&self.tensors[a_id].data, &self.tensors[out_id].data) {
                        (Storage::Gpu(a_s), Storage::Gpu(o_s)) => {
                            builder.arg(a_s).arg(o_s)
                                .arg(&s_u64).arg(&hd_u64).arg(&hdim_u64).arg(&np_u64);
                        }
                        #[cfg(feature = "bf16")]
                        (Storage::GpuBf16(a_s), Storage::GpuBf16(o_s)) => {
                            builder.arg(a_s).arg(o_s)
                                .arg(&s_u64).arg(&hd_u64).arg(&hdim_u64).arg(&np_u64);
                        }
                        _ => panic!("rope: mismatched storage"),
                    }
                    unsafe { builder.launch(LaunchConfig::for_num_elems(num_pairs as u32)) }.unwrap();
                }

                // rope_backward recomputes cos/sin — no saved activation, FP32 grad only
                let backward_fn = Box::new(move |tensors: &mut [Tensor]| {
                    let out_grad = match &tensors[out_id].grad {
                        Storage::Gpu(s) => s, _ => unreachable!()
                    };
                    let a_grad = match &tensors[a_id].grad {
                        Storage::Gpu(s) => s, _ => unreachable!()
                    };
                    let mut b1 = stream_clone.launch_builder(&f_bwd);
                    b1.arg(out_grad).arg(a_grad)
                        .arg(&s_u64).arg(&hd_u64).arg(&hdim_u64).arg(&np_u64);
                    unsafe { b1.launch(LaunchConfig::for_num_elems(num_pairs as u32)) }.unwrap();
                });

                if !self.no_grad {
                    self.tape.nodes.push(TapeNode {
                        inputs: vec![a_id], output: out_id, backward_fn,
                    });
                }
                out_id
            }
            Device::Cpu => {
                let a_data = self.tensors[a_id].data.as_cpu().clone();
                let batch = shape[0];
                let mut out_data = a_data.clone();

                for bb in 0..batch {
                    for pos in 0..seq_len {
                        for base in (0..hidden_dim).step_by(head_dim) {
                            let pair_limit = head_dim / 2;
                            for pair in 0..pair_limit {
                                let even = base + 2 * pair;
                                let odd = even + 1;
                                if odd >= base + head_dim || odd >= hidden_dim {
                                    continue;
                                }
                                let idx0 = ((bb * seq_len + pos) * hidden_dim) + even;
                                let idx1 = ((bb * seq_len + pos) * hidden_dim) + odd;

                                let theta = (pos as f32)
                                    / 10000_f32.powf((2.0 * pair as f32) / (head_dim as f32));
                                let (sin_t, cos_t) = theta.sin_cos();
                                let x0 = a_data[idx0];
                                let x1 = a_data[idx1];
                                out_data[idx0] = x0 * cos_t - x1 * sin_t;
                                out_data[idx1] = x0 * sin_t + x1 * cos_t;
                            }
                        }
                    }
                }

                let out_id = self.alloc(shape, out_data);
                let backward_fn = Box::new(move |tensors: &mut [Tensor]| {
                    let out_grad = tensors[out_id].grad.as_cpu().clone();
                    let a_grad = tensors[a_id].grad.as_cpu_mut();

                    for bb in 0..batch {
                        for pos in 0..seq_len {
                            for base in (0..hidden_dim).step_by(head_dim) {
                                let pair_limit = head_dim / 2;
                                for pair in 0..pair_limit {
                                    let even = base + 2 * pair;
                                    let odd = even + 1;
                                    if odd >= base + head_dim || odd >= hidden_dim {
                                        continue;
                                    }
                                    let idx0 = ((bb * seq_len + pos) * hidden_dim) + even;
                                    let idx1 = ((bb * seq_len + pos) * hidden_dim) + odd;

                                    let theta = (pos as f32)
                                        / 10000_f32.powf((2.0 * pair as f32) / (head_dim as f32));
                                    let (sin_t, cos_t) = theta.sin_cos();
                                    let g0 = out_grad[idx0];
                                    let g1 = out_grad[idx1];
                                    a_grad[idx0] += g0 * cos_t + g1 * sin_t;
                                    a_grad[idx1] += -g0 * sin_t + g1 * cos_t;
                                }
                            }
                        }
                    }
                });

                if !self.no_grad {
                    self.tape.nodes.push(TapeNode {
                        inputs: vec![a_id],
                        output: out_id,
                        backward_fn,
                    });
                }
                out_id
            }
        }
    }

}
