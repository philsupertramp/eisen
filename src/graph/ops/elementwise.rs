use crate::graph::{Graph, TapeNode, is_bf16};
use crate::tensor::{Tensor, Device, Storage};
use cudarc::driver::{PushKernelArg, LaunchConfig, CudaSlice, CudaFunction};


impl Graph {
    pub fn add(&mut self, a_id: usize, b_id: usize) -> usize {
        let a_shape = self.tensors[a_id].shape.clone();
        let b_shape = self.tensors[b_id].shape.clone();
        let out_shape = Tensor::get_broadcasted_shape(&a_shape, &b_shape);
        let out_size = out_shape.iter().product::<usize>();
        let device = self.device.clone();

        match &device {
            Device::Gpu(_, stream) => {
                #[allow(unused_variables)]
                let bf16_mode = self.uses_bf16_mixed_precision();

                let a_strides = Tensor::get_broadcasted_strides(
                    &a_shape, &self.tensors[a_id].strides, &out_shape);
                let b_strides = Tensor::get_broadcasted_strides(
                    &b_shape, &self.tensors[b_id].strides, &out_shape);
                let rank = out_shape.len() as u64;

                let mut s    = [1u64; 3];
                let mut a_str = [0u64; 3];
                let mut b_str = [0u64; 3];
                for i in 0..out_shape.len() {
                    s[i]     = out_shape[i] as u64;
                    a_str[i] = a_strides[i] as u64;
                    b_str[i] = b_strides[i] as u64;
                }

                let n = out_size as u64;
                let stream_clone = stream.clone();

                // Kernel selection: BF16×BF16→BF16,  BF16×FP32→BF16,  FP32×FP32→FP32
                #[cfg(feature = "bf16")]
                let (f_fwd, _f_accumulate_bwd) = {
                    let a_is_bf16 = is_bf16(&self.tensors[a_id].data);
                    let b_is_bf16 = is_bf16(&self.tensors[b_id].data);
                    match (bf16_mode, a_is_bf16, b_is_bf16) {
                        (true, true, true)  => (
                            self.functions.get("add_bf16").unwrap().clone(),
                            None::<CudaFunction>, // backward uses accumulate_f32 (grads FP32)
                        ),
                        (true, true, false) => (
                            self.functions.get("add_bf16lhs_f32rhs_bf16out").unwrap().clone(),
                            None,
                        ),
                        _ => (
                            self.functions.get("add_f32").unwrap().clone(),
                            None,
                        ),
                    }
                };
                #[cfg(not(feature = "bf16"))]
                let (f_fwd, _) = (self.functions.get("add_f32").unwrap().clone(), ());

                let f_accumulate = self.functions.get("accumulate_f32").unwrap().clone();
                let b_u16_slice: CudaSlice<u16>;

                let out_id = self.alloc_pooled(out_shape.clone());
                self.name_tensor(out_id, "add_output");

                {
                    let mut builder = stream.launch_builder(&f_fwd);
                    // push stride args after data args — same shape regardless of kernel variant
                    macro_rules! push_stride_args {
                        ($builder:ident) => {
                            $builder.arg(&n).arg(&rank)
                                .arg(&s[0]).arg(&s[1]).arg(&s[2])
                                .arg(&a_str[0]).arg(&a_str[1]).arg(&a_str[2])
                                .arg(&b_str[0]).arg(&b_str[1]).arg(&b_str[2]);
                        };
                    }

                    match (
                        &self.tensors[a_id].data,
                        &self.tensors[b_id].data,
                        &self.tensors[out_id].data,
                    ) {
                        (Storage::Gpu(a), Storage::Gpu(b), Storage::Gpu(o)) => {
                            builder.arg(a).arg(b).arg(o);
                            push_stride_args!(builder);
                        }
                        #[cfg(feature = "bf16")]
                        (Storage::GpuBf16(a), Storage::GpuBf16(b), Storage::GpuBf16(o)) => {
                            builder.arg(a).arg(b).arg(o);
                            push_stride_args!(builder);
                        }
                        #[cfg(feature = "bf16")]
                        (Storage::GpuBf16(a), Storage::Gpu(b), Storage::GpuBf16(o)) => {
                            builder.arg(a).arg(b).arg(o);
                            push_stride_args!(builder);
                        }
                        #[cfg(feature = "bf16")]
                        (Storage::GpuBf16(a), Storage::Cpu(b), Storage::GpuBf16(o)) => {
                            let stream = match &self.device { Device::Gpu(_, s) => s.clone(), _ => unreachable!() };
                            b_u16_slice = stream
                                .clone_htod(&b.as_slice().iter().map(|&x| x as u16).collect::<Vec<u16>>())
                                .expect("bf16_streamed: forward add failed");

                            builder.arg(a).arg(&b_u16_slice).arg(o);
                            push_stride_args!(builder);
                        }
                        (p1, p2, p3) => panic!(
                            "add: unsupported storage combination. Received: ({:?}, {:?}, {:?})",
                            p1, p2, p3
                        ),
                    }
                    unsafe { builder.launch(LaunchConfig::for_num_elems(out_size as u32)) }.unwrap();
                }

                // Backward: grads are always FP32 — use accumulate_f32 unchanged
                let backward_fn = Box::new(move |tensors: &mut [Tensor]| {
                    let out_grad = match &tensors[out_id].grad {
                        Storage::Gpu(s) => s, _ => unreachable!()
                    };
                    let a_grad = match &tensors[a_id].grad {
                        Storage::Gpu(s) => s, _ => unreachable!()
                    };
                    let b_grad = match &tensors[b_id].grad {
                        Storage::Gpu(s) => s, _ => unreachable!()
                    };

                    // accumulate_f32 signature: (grad_target, grad_out, n, rank, s*, t*)
                    let launch_accum = |tgt: &CudaSlice<f32>, t_str: [u64; 3]| {
                        let mut b1 = stream_clone.launch_builder(&f_accumulate);
                        b1.arg(tgt).arg(out_grad).arg(&n).arg(&rank)
                            .arg(&s[0]).arg(&s[1]).arg(&s[2])
                            .arg(&t_str[0]).arg(&t_str[1]).arg(&t_str[2]);
                        unsafe { b1.launch(LaunchConfig::for_num_elems(out_size as u32)) }.unwrap();
                    };
                    launch_accum(a_grad, a_str);
                    launch_accum(b_grad, b_str);
                });

                if !self.no_grad {
                    self.tape.nodes.push(TapeNode {
                        inputs: vec![a_id, b_id], output: out_id, backward_fn,
                    });
                }
                out_id
            }
            Device::Cpu => {
                let a_strides = Tensor::get_broadcasted_strides(
                    &a_shape,
                    &self.tensors[a_id].strides,
                    &out_shape,
                );
                let b_strides = Tensor::get_broadcasted_strides(
                    &b_shape,
                    &self.tensors[b_id].strides,
                    &out_shape,
                );
                let mut out_data = vec![0.0; out_size];
                let a_data = self.tensors[a_id].data.as_cpu();
                let b_data = self.tensors[b_id].data.as_cpu();
                for i in 0..out_size {
                    let nd = Tensor::flat_to_nd(i, &out_shape);
                    out_data[i] = a_data[Tensor::nd_to_flat(&nd, &a_strides)]
                        + b_data[Tensor::nd_to_flat(&nd, &b_strides)];
                }
                let out_id = self.alloc(out_shape.clone(), out_data);
                let backward_fn = Box::new(move |tensors: &mut [Tensor]| {
                    let out_grad = tensors[out_id].grad.as_cpu().clone();
                    for i in 0..out_size {
                        let nd = Tensor::flat_to_nd(i, &out_shape);
                        tensors[a_id].grad.as_cpu_mut()[Tensor::nd_to_flat(&nd, &a_strides)] +=
                            out_grad[i];
                        tensors[b_id].grad.as_cpu_mut()[Tensor::nd_to_flat(&nd, &b_strides)] +=
                            out_grad[i];
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

    pub fn mul(&mut self, a_id: usize, b_id: usize) -> usize {
        let a_shape  = self.tensors[a_id].shape.clone();
        let b_shape  = self.tensors[b_id].shape.clone();
        let out_shape = Tensor::get_broadcasted_shape(&a_shape, &b_shape);
        let out_size: usize = out_shape.iter().product();
        let device = self.device.clone();

        match &device {
            Device::Gpu(_, stream) => {
                let stream_clone = stream.clone();

                // Select kernels based on storage combination
                #[cfg(feature = "bf16")]
                let (f_fwd, f_bwd) = {
                    let a_bf = is_bf16(&self.tensors[a_id].data);
                    let b_bf = is_bf16(&self.tensors[b_id].data);
                    match (a_bf, b_bf) {
                        (true, true)  => (
                            self.functions.get("mul_bf16").unwrap().clone(),
                            self.functions.get("mul_backward_bf16in_f32").unwrap().clone(),
                        ),
                        (true, false) => (
                            self.functions.get("mul_bf16lhs_f32rhs_bf16out").unwrap().clone(),
                            self.functions.get("mul_backward_bf16lhs_f32rhs").unwrap().clone(),
                        ),
                        _ => (
                            self.functions.get("mul_f32").unwrap().clone(),
                            self.functions.get("mul_backward_f32").unwrap().clone(),
                        ),
                    }
                };
                #[cfg(not(feature = "bf16"))]
                let (f_fwd, f_bwd) = (
                    self.functions.get("mul_f32").unwrap().clone(),
                    self.functions.get("mul_backward_f32").unwrap().clone(),
                );

                let out_id = self.alloc_pooled(out_shape);
                self.name_tensor(out_id, "mul_output");

                let n = out_size as u64;

                {
                    let mut builder = stream.launch_builder(&f_fwd);
                    match (
                        &self.tensors[a_id].data,
                        &self.tensors[b_id].data,
                        &self.tensors[out_id].data,
                    ) {
                        (Storage::Gpu(a), Storage::Gpu(b), Storage::Gpu(o)) => {
                            builder.arg(a).arg(b).arg(o).arg(&n);
                        }
                        #[cfg(feature = "bf16")]
                        (Storage::GpuBf16(a), Storage::GpuBf16(b), Storage::GpuBf16(o)) => {
                            builder.arg(a).arg(b).arg(o).arg(&n);
                        }
                        #[cfg(feature = "bf16")]
                        (Storage::GpuBf16(a), Storage::Gpu(b), Storage::GpuBf16(o)) => {
                            builder.arg(a).arg(b).arg(o).arg(&n);
                        }
                        (p1, p2, p3) => panic!(
                            "mul: unsupported storage combination. Received: ({:?}, {:?}, {:?})",
                            p1, p2, p3
                        ),
                    }
                    unsafe { builder.launch(LaunchConfig::for_num_elems(out_size as u32)) }.unwrap();
                }

                let backward_fn = Box::new(move |tensors: &mut [Tensor]| {
                    let out_grad = match &tensors[out_id].grad {
                        Storage::Gpu(s) => s, _ => unreachable!()
                    };
                    let a_grad = match &tensors[a_id].grad {
                        Storage::Gpu(s) => s, _ => unreachable!()
                    };
                    let b_grad = match &tensors[b_id].grad {
                        Storage::Gpu(s) => s, _ => unreachable!()
                    };
                    let mut builder = stream_clone.launch_builder(&f_bwd);
                    match (&tensors[a_id].data, &tensors[b_id].data) {
                        (Storage::Gpu(a), Storage::Gpu(b)) => {
                            builder.arg(a).arg(b).arg(out_grad).arg(a_grad).arg(b_grad).arg(&n);
                        }
                        #[cfg(feature = "bf16")]
                        (Storage::GpuBf16(a), Storage::GpuBf16(b)) => {
                            builder.arg(a).arg(b).arg(out_grad).arg(a_grad).arg(b_grad).arg(&n);
                        }
                        #[cfg(feature = "bf16")]
                        (Storage::GpuBf16(a), Storage::Gpu(b)) => {
                            builder.arg(a).arg(b).arg(out_grad).arg(a_grad).arg(b_grad).arg(&n);
                        }
                        (p1, p2) => panic!(
                            "mul_backward: unsupported storage combination. Received: ({:?}, {:?})",
                            p1, p2
                        ),
                    }
                    unsafe { builder.launch(LaunchConfig::for_num_elems(out_size as u32)) }.unwrap();
                });

                if !self.no_grad {
                    self.tape.nodes.push(TapeNode {
                        inputs: vec![a_id, b_id], output: out_id, backward_fn,
                    });
                }
                out_id
            }
            Device::Cpu => {
                let a_strides = Tensor::get_broadcasted_strides(
                    &a_shape,
                    &self.tensors[a_id].strides,
                    &out_shape,
                );
                let b_strides = Tensor::get_broadcasted_strides(
                    &b_shape,
                    &self.tensors[b_id].strides,
                    &out_shape,
                );
                let mut out_data = vec![0.0; out_size];
                let a_fwd = self.tensors[a_id].data.as_cpu().clone();
                let b_fwd = self.tensors[b_id].data.as_cpu().clone();
                for i in 0..out_size {
                    let nd = Tensor::flat_to_nd(i, &out_shape);
                    out_data[i] = a_fwd[Tensor::nd_to_flat(&nd, &a_strides)]
                        * b_fwd[Tensor::nd_to_flat(&nd, &b_strides)];
                }
                let out_id = self.alloc(out_shape.clone(), out_data);
                let backward_fn = Box::new(move |tensors: &mut [Tensor]| {
                    let out_grad = tensors[out_id].grad.as_cpu().clone();
                    for i in 0..out_size {
                        let nd = Tensor::flat_to_nd(i, &out_shape);
                        tensors[a_id].grad.as_cpu_mut()[Tensor::nd_to_flat(&nd, &a_strides)] +=
                            b_fwd[Tensor::nd_to_flat(&nd, &b_strides)] * out_grad[i];
                        tensors[b_id].grad.as_cpu_mut()[Tensor::nd_to_flat(&nd, &b_strides)] +=
                            a_fwd[Tensor::nd_to_flat(&nd, &a_strides)] * out_grad[i];
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


    pub fn silu(&mut self, a_id: usize) -> usize {
        let device = self.device.clone();
        match &device {
            Device::Gpu(_, stream) => {
                let a_shape    = self.tensors[a_id].shape.clone();
                let out_size   = a_shape.iter().product::<usize>();
                #[allow(unused_variables)]
                let bf16_mode  = self.uses_bf16_mixed_precision();
                let stream_clone = stream.clone();

                // ── Select forward kernel ──────────────────────────────────────
                #[cfg(feature = "bf16")]
                let (f_fwd, f_bwd) = if bf16_mode {
                    (
                        self.functions.get("silu_bf16").unwrap().clone(),
                        self.functions.get("silu_backward_bf16in_f32").unwrap().clone(),
                    )
                } else {
                    (
                        self.functions.get("silu_f32").unwrap().clone(),
                        self.functions.get("silu_backward_f32").unwrap().clone(),
                    )
                };
                #[cfg(not(feature = "bf16"))]
                let (f_fwd, f_bwd) = (
                    self.functions.get("silu_f32").unwrap().clone(),
                    self.functions.get("silu_backward_f32").unwrap().clone(),
                );

                let out_id = self.alloc_pooled(a_shape);
                self.name_tensor(out_id, "silu_output");

                // ── Forward launch ─────────────────────────────────────────────
                {
                    let n = out_size as u64;
                    let mut builder = stream.launch_builder(&f_fwd);
                    // Dispatch based on storage tags
                    match (&self.tensors[a_id].data, &self.tensors[out_id].data) {
                        (Storage::Gpu(a_s), Storage::Gpu(o_s)) => {
                            builder.arg(a_s).arg(o_s).arg(&n);
                        }
                        #[cfg(feature = "bf16")]
                        (Storage::GpuBf16(a_s), Storage::GpuBf16(o_s)) => {
                            builder.arg(a_s).arg(o_s).arg(&n);
                        }
                        _ => panic!("silu: mismatched storage types"),
                    }
                    unsafe { builder.launch(LaunchConfig::for_num_elems(out_size as u32)) }.unwrap();
                }

                // ── Capture kernel handles for backward ────────────────────────
                #[cfg(feature = "bf16")]
                let f_cast = if bf16_mode {
                    Some(self.functions.get("cast_bf16_to_f32").unwrap().clone())
                } else { None };
                #[cfg(not(feature = "bf16"))]
                #[allow(unused_variables)]
                let _f_cast: Option<()> = None;

                let backward_fn = Box::new(move |tensors: &mut [Tensor]| {
                    let n = out_size as u64;
                    let out_grad = match &tensors[out_id].grad {
                        Storage::Gpu(s) => s, _ => unreachable!()
                    };
                    let a_grad = match &tensors[a_id].grad {
                        Storage::Gpu(s) => s, _ => unreachable!()
                    };

                    // If activation stored as BF16, cast to FP32 temp for backward kernel
                    #[cfg(feature = "bf16")]
                    let a_temp = f_cast.as_ref().and_then(|fc| {
                        crate::bf16_util::bf16_to_f32_temp(
                            &tensors[a_id].data, out_size, &stream_clone, fc,
                        )
                    });
                    #[cfg(not(feature = "bf16"))]
                    let a_temp: Option<()> = None;

                    let mut builder = stream_clone.launch_builder(&f_bwd);
                    match (&tensors[a_id].data, &a_temp) {
                        #[cfg(feature = "bf16")]
                        (_, Some(t)) => {
                            // silu_backward_bf16in_f32 already consumed — but we're in the
                            // FP32 backward kernel path after the cast
                            builder.arg(t).arg(out_grad).arg(a_grad).arg(&n);
                        }
                        (Storage::Gpu(s), _) => {
                            builder.arg(s).arg(out_grad).arg(a_grad).arg(&n);
                        }
                        _ => unreachable!(),
                    }
                    unsafe { builder.launch(LaunchConfig::for_num_elems(out_size as u32)) }.unwrap();
                    // a_temp dropped here → cudaFree
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
                let mut out_data = vec![0.0; a_data.len()];
                for i in 0..a_data.len() {
                    let x = a_data[i];
                    out_data[i] = x * (1.0 / (1.0 + (-x).exp()));
                }
                let out_id = self.alloc(self.tensors[a_id].shape.clone(), out_data);
                let backward_fn = Box::new(move |tensors: &mut [Tensor]| {
                    let out_grad = tensors[out_id].grad.as_cpu().clone();
                    let a_grad = tensors[a_id].grad.as_cpu_mut();
                    for i in 0..a_grad.len() {
                        let x = a_data[i];
                        let sig = 1.0 / (1.0 + (-x).exp());
                        let silu = x * sig;
                        a_grad[i] += out_grad[i] * (silu + sig * (1.0 - silu));
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
