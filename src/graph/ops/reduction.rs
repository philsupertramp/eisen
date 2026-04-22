use crate::graph::{Graph, TapeNode, is_bf16};
use crate::tensor::{Tensor, Device, Storage};
use crate::data::fim::IGNORE_INDEX;
use cudarc::driver::{PushKernelArg, LaunchConfig};
use crate::safe_bf16_temp;

impl Graph {

    pub fn sum(&mut self, a_id: usize, dim: usize) -> usize {
        let device = self.device.clone();
        match &device {
            Device::Gpu(_, stream) => {
                let a = &self.tensors[a_id];
                let _a_size = a.shape.iter().product::<usize>();
                let mut out_shape = a.shape.clone();
                let mut in_strides = a.strides.clone();
                let reduced_dim_size = a.shape[dim] as u64;
                let reduced_dim_stride = a.strides[dim] as u64;
                out_shape.remove(dim);
                in_strides.remove(dim);
                let out_size = if out_shape.is_empty() {
                    1
                } else {
                    out_shape.iter().product()
                };
                let rank = out_shape.len() as u64;

                let mut os = [1u64; 3];
                let mut is = [0u64; 3];
                for i in 0..out_shape.len() {
                    os[i] = out_shape[i] as u64;
                    is[i] = in_strides[i] as u64;
                }

                let f_fwd = self.functions.get("sum_f32").unwrap().clone();
                let f_bwd = self.functions.get("sum_backward_f32").unwrap().clone();
                let stream_clone = stream.clone();

                let out_id = self.alloc_pooled(out_shape.clone());
                self.name_tensor(out_id, "sum_output");

                #[cfg(feature = "bf16")]
                let (compute_target_id, cast_to_bf16_after) = if self.uses_bf16_mixed_precision() {
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
                let a_temp_fwd = safe_bf16_temp!(self, a_id, _a_size, stream, &f_cast_to_f32);
                #[cfg(not(feature = "bf16"))]
                let a_temp_fwd: Option<()> = None;

                let a_s = match (&self.tensors[a_id].data, &a_temp_fwd) {
                    (Storage::Gpu(s), _) => s,
                    #[cfg(feature = "bf16")] (_, Some(t)) => t,
                    _ => unreachable!("sum: input must be Gpu or GpuBf16"),
                };
                let o_s = match &self.tensors[compute_target_id].data {
                    Storage::Gpu(s) => s,
                    _ => unreachable!(),
                };

                let out_size_u64 = out_size as u64;
                let mut builder = stream.launch_builder(&f_fwd);
                builder
                    .arg(a_s)
                    .arg(o_s)
                    .arg(&out_size_u64)
                    .arg(&reduced_dim_size)
                    .arg(&reduced_dim_stride)
                    .arg(&rank)
                    .arg(&os[0])
                    .arg(&os[1])
                    .arg(&os[2])
                    .arg(&is[0])
                    .arg(&is[1])
                    .arg(&is[2]);
                unsafe { builder.launch(LaunchConfig::for_num_elems(out_size as u32)) }.unwrap();

                let backward_fn = Box::new(move |tensors: &mut [Tensor]| {
                    let out_grad = match &tensors[out_id].grad {
                        Storage::Gpu(s) => s,
                        _ => unreachable!("sum backward: out_grad must be Gpu"),
                    };
                    let a_grad = match &tensors[a_id].grad {
                        Storage::Gpu(s) => s,
                        _ => unreachable!("sum backward: a_grad must be Gpu"),
                    };

                    let mut b1 = stream_clone.launch_builder(&f_bwd);
                    b1.arg(out_grad)
                        .arg(a_grad)
                        .arg(&out_size_u64)
                        .arg(&reduced_dim_size)
                        .arg(&reduced_dim_stride)
                        .arg(&rank)
                        .arg(&os[0])
                        .arg(&os[1])
                        .arg(&os[2])
                        .arg(&is[0])
                        .arg(&is[1])
                        .arg(&is[2]);
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
                        inputs: vec![a_id],
                        output: out_id,
                        backward_fn,
                    });
                }
                out_id
            }
            Device::Cpu => {
                let a = &self.tensors[a_id];
                let mut out_shape = a.shape.clone();
                out_shape.remove(dim);
                let out_size = if out_shape.is_empty() {
                    1
                } else {
                    out_shape.iter().product()
                };
                let mut out_data = vec![0.0; out_size];
                let os = Tensor::compute_strides(&out_shape);
                let a_shape = a.shape.clone();
                let a_data = a.data.as_cpu();
                for i in 0..a_data.len() {
                    let mut nd = Tensor::flat_to_nd(i, &a_shape);
                    nd.remove(dim);
                    let idx = if out_shape.is_empty() {
                        0
                    } else {
                        Tensor::nd_to_flat(&nd, &os)
                    };
                    out_data[idx] += a_data[i];
                }
                let out_id = self.alloc(out_shape, out_data);
                let backward_fn = Box::new(move |tensors: &mut [Tensor]| {
                    let o_grad = tensors[out_id].grad.as_cpu().clone();
                    for i in 0..tensors[a_id].data.as_cpu().len() {
                        let mut nd = Tensor::flat_to_nd(i, &a_shape);
                        nd.remove(dim);
                        let idx = if tensors[out_id].shape.is_empty() {
                            0
                        } else {
                            Tensor::nd_to_flat(&nd, &os)
                        };
                        tensors[a_id].grad.as_cpu_mut()[i] += o_grad[idx];
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

    pub fn max(&mut self, a_id: usize, dim: usize) -> usize {
        let device = self.device.clone();
        match &device {
            Device::Gpu(_, stream) => {
                let a = &self.tensors[a_id];
                let _a_size = a.shape.iter().product::<usize>();
                let mut out_shape = a.shape.clone();
                let mut in_strides = a.strides.clone();
                let reduced_dim_size = a.shape[dim] as u64;
                let reduced_dim_stride = a.strides[dim] as u64;
                out_shape.remove(dim);
                in_strides.remove(dim);
                let out_size = if out_shape.is_empty() {
                    1
                } else {
                    out_shape.iter().product()
                };
                let rank = out_shape.len() as u64;

                let mut os = [1u64; 3];
                let mut is = [0u64; 3];
                for i in 0..out_shape.len() {
                    os[i] = out_shape[i] as u64;
                    is[i] = in_strides[i] as u64;
                }

                let f_fwd = self.functions.get("max_f32").unwrap().clone();
                let f_bwd = self.functions.get("max_backward_f32").unwrap().clone();
                let stream_clone = stream.clone();

                let out_id = self.alloc_pooled(out_shape.clone());
                self.name_tensor(out_id, "max_output");

                #[cfg(feature = "bf16")]
                let (compute_target_id, cast_to_bf16_after) = if self.uses_bf16_mixed_precision() {
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
                let a_temp_fwd = safe_bf16_temp!(self, a_id, _a_size, stream, &f_cast_to_f32);
                #[cfg(not(feature = "bf16"))]
                let a_temp_fwd: Option<()> = None;

                let a_s = match (&self.tensors[a_id].data, &a_temp_fwd) {
                    (Storage::Gpu(s), _) => s,
                    #[cfg(feature = "bf16")] (_, Some(t)) => t,
                    _ => unreachable!("max: input must be Gpu or GpuBf16"),
                };
                let o_s = match &self.tensors[compute_target_id].data {
                    Storage::Gpu(s) => s,
                    _ => unreachable!(),
                };

                let out_size_u64 = out_size as u64;
                let mut builder = stream.launch_builder(&f_fwd);
                builder
                    .arg(a_s)
                    .arg(o_s)
                    .arg(&out_size_u64)
                    .arg(&reduced_dim_size)
                    .arg(&reduced_dim_stride)
                    .arg(&rank)
                    .arg(&os[0])
                    .arg(&os[1])
                    .arg(&os[2])
                    .arg(&is[0])
                    .arg(&is[1])
                    .arg(&is[2]);
                unsafe { builder.launch(LaunchConfig::for_num_elems(out_size as u32)) }.unwrap();

                let backward_fn = Box::new(move |tensors: &mut [Tensor]| {
                    #[cfg(feature = "bf16")]
                    let a_temp_bwd = crate::bf16_util::bf16_to_f32_temp(&tensors[a_id].data, _a_size, &stream_clone, &f_cast_to_f32_bwd);
                    #[cfg(not(feature = "bf16"))]
                    let a_temp_bwd: Option<()> = None;

                    let a_data = match (&tensors[a_id].data, &a_temp_bwd) {
                        (Storage::Gpu(s), _) => s,
                        #[cfg(feature = "bf16")] (_, Some(t)) => t,
                        _ => unreachable!("max backward: input must be Gpu or GpuBf16"),
                    };

                    let out_grad = match &tensors[out_id].grad {
                        Storage::Gpu(s) => s,
                        _ => unreachable!("max backward: out_grad must be Gpu"),
                    };
                    let a_grad = match &tensors[a_id].grad {
                        Storage::Gpu(s) => s,
                        _ => unreachable!("max backward: a_grad must be Gpu"),
                    };

                    let mut b1 = stream_clone.launch_builder(&f_bwd);
                    b1.arg(a_data)
                        .arg(out_grad)
                        .arg(a_grad)
                        .arg(&out_size_u64)
                        .arg(&reduced_dim_size)
                        .arg(&reduced_dim_stride)
                        .arg(&rank)
                        .arg(&os[0])
                        .arg(&os[1])
                        .arg(&os[2])
                        .arg(&is[0])
                        .arg(&is[1])
                        .arg(&is[2]);
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
                        inputs: vec![a_id],
                        output: out_id,
                        backward_fn,
                    });
                }
                out_id
            }
            Device::Cpu => {
                let a = &self.tensors[a_id];
                let mut out_shape = a.shape.clone();
                out_shape.remove(dim);
                let out_size = if out_shape.is_empty() {
                    1
                } else {
                    out_shape.iter().product()
                };
                let mut out_data = vec![f32::NEG_INFINITY; out_size];
                let mut argmax = vec![0; out_size];
                let os = Tensor::compute_strides(&out_shape);
                let a_data = a.data.as_cpu();
                for i in 0..a_data.len() {
                    let mut nd = Tensor::flat_to_nd(i, &a.shape);
                    nd.remove(dim);
                    let idx = if out_shape.is_empty() {
                        0
                    } else {
                        Tensor::nd_to_flat(&nd, &os)
                    };
                    if a_data[i] > out_data[idx] {
                        out_data[idx] = a_data[i];
                        argmax[idx] = i;
                    }
                }
                let out_id = self.alloc(out_shape, out_data);
                let backward_fn = Box::new(move |tensors: &mut [Tensor]| {
                    let o_grad = tensors[out_id].grad.as_cpu().clone();
                    for i in 0..o_grad.len() {
                        tensors[a_id].grad.as_cpu_mut()[argmax[i]] += o_grad[i];
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

    pub fn softmax(&mut self, a_id: usize) -> usize {
        let device = self.device.clone();
        match &device {
            Device::Gpu(_, stream) => {
                let a = &self.tensors[a_id];
                let out_shape = a.shape.clone();
                let n  = *out_shape.last().unwrap() as u64;
                let b  = (a.data.len() as u64) / n;
                let stream_clone = stream.clone();

                #[cfg(feature = "bf16")]
                let (f_fwd, f_bwd) = if is_bf16(&self.tensors[a_id].data) {
                    (
                        self.functions.get("softmax_bf16").unwrap().clone(),
                        self.functions.get("softmax_backward_bf16in_f32").unwrap().clone(),
                    )
                } else {
                    (
                        self.functions.get("softmax_f32").unwrap().clone(),
                        self.functions.get("softmax_backward_f32").unwrap().clone(),
                    )
                };
                #[cfg(not(feature = "bf16"))]
                let (f_fwd, f_bwd) = (
                    self.functions.get("softmax_f32").unwrap().clone(),
                    self.functions.get("softmax_backward_f32").unwrap().clone(),
                );

                let out_id = self.alloc_pooled(out_shape);
                self.name_tensor(out_id, "softmax_output");

                self.ensure_on_gpu(a_id);
                self.ensure_on_gpu(out_id);

                {
                    let mut builder = stream.launch_builder(&f_fwd);
                    match (&self.tensors[a_id].data, &self.tensors[out_id].data) {
                        (Storage::Gpu(a_s), Storage::Gpu(o_s)) => {
                            builder.arg(a_s).arg(o_s).arg(&b).arg(&n);
                        }
                        #[cfg(feature = "bf16")]
                        (Storage::GpuBf16(a_s), Storage::GpuBf16(o_s)) => {
                            builder.arg(a_s).arg(o_s).arg(&b).arg(&n);
                        }
                        (p1, p2) => panic!(
                            "softmax: unsupported storage combination. Received: ({:?}, {:?})",
                            p1, p2
                        ),
                    }
                    unsafe { builder.launch(LaunchConfig::for_num_elems(b as u32)) }.unwrap();
                }

                let backward_fn = Box::new(move |tensors: &mut [Tensor]| {
                    let out_grad = match &tensors[out_id].grad {
                        Storage::Gpu(s) => s, _ => unreachable!()
                    };
                    let a_grad = match &tensors[a_id].grad {
                        Storage::Gpu(s) => s, _ => unreachable!()
                    };
                    let mut builder = stream_clone.launch_builder(&f_bwd);
                    // softmax_backward_bf16in_f32: (bf16 out_data, f32 grad_out, f32 grad_x, B, N)
                    // softmax_backward_f32:        (f32  out_data, f32 grad_out, f32 grad_x, B, N)
                    match &tensors[out_id].data {
                        Storage::Gpu(out_data) => {
                            builder.arg(out_data).arg(out_grad).arg(a_grad).arg(&b).arg(&n);
                        }
                        #[cfg(feature = "bf16")]
                        Storage::GpuBf16(out_data) => {
                            builder.arg(out_data).arg(out_grad).arg(a_grad).arg(&b).arg(&n);
                        }
                        _ => unreachable!(),
                    }
                    unsafe { builder.launch(LaunchConfig::for_num_elems(b as u32)) }.unwrap();
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
                let a_shape = self.tensors[a_id].shape.clone();
                let n = *a_shape.last().unwrap();
                let b = a_data.len() / n;
                let mut out_data = vec![0.0; a_data.len()];

                for row in 0..b {
                    let off = row * n;
                    let row_slice = &a_data[off..off + n];
                    let max_v = row_slice.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let mut sum_exp = 0.0;
                    for j in 0..n {
                        let e = (row_slice[j] - max_v).exp();
                        out_data[off + j] = e;
                        sum_exp += e;
                    }
                    for j in 0..n {
                        out_data[off + j] /= sum_exp;
                    }
                }

                let out_fwd = out_data.clone();
                let out_id = self.alloc(a_shape, out_data);
                let backward_fn = Box::new(move |tensors: &mut [Tensor]| {
                    let out_grad = tensors[out_id].grad.as_cpu().clone();
                    let a_grad = tensors[a_id].grad.as_cpu_mut();

                    for row in 0..b {
                        let off = row * n;
                        let mut dot = 0.0;
                        for j in 0..n {
                            dot += out_grad[off + j] * out_fwd[off + j];
                        }
                        for j in 0..n {
                            a_grad[off + j] += out_fwd[off + j] * (out_grad[off + j] - dot);
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


    pub fn cross_entropy(&mut self, logits_id: usize, targets: &[usize]) -> usize {
        let device = self.device.clone();
        match &device {
            Device::Gpu(_, stream) => {
                let logits = &self.tensors[logits_id];
                let batch_size = logits.shape[0];
                let num_classes = logits.shape[1];
                let out_id = self.alloc_pooled(vec![]);
                self.name_tensor(out_id, "cross_entropy_output");
                #[cfg(feature = "bf16")]
                let (compute_target_id, cast_to_bf16_after) = if self.uses_bf16_mixed_precision() {
                    let f32_slice = self.safe_alloc_zeros::<f32>(stream, 1);
                    let tmp_id = self.tensors.len();
                    let grad_slice = self.safe_alloc_zeros::<f32>(stream, 1);
                    self.tensors.push(Tensor {
                        id: tmp_id, shape: vec![],
                        strides: vec![],
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

                let f_fwd = self.functions.get("cross_entropy_f32").unwrap().clone();
                let f_bwd = self
                    .functions
                    .get("cross_entropy_backward_f32")
                    .unwrap()
                    .clone();
                let stream_clone = stream.clone();

                let targets_f32: Vec<f32> = targets.iter().map(|&x| x as f32).collect();
                let targets_d = stream.clone_htod(targets_f32.as_slice()).unwrap();

                #[cfg(feature = "bf16")]
                let f_cast_to_f32 = self.functions.get("cast_bf16_to_f32").unwrap().clone();
                #[cfg(feature = "bf16")]
                let f_cast_to_f32_bwd = f_cast_to_f32.clone();

                #[cfg(feature = "bf16")]
                let l_temp_fwd = safe_bf16_temp!(self, logits_id, batch_size * num_classes, stream, &f_cast_to_f32);
                #[cfg(not(feature = "bf16"))]
                let l_temp_fwd: Option<()> = None;

                if let (Storage::Gpu(s), Device::Gpu(_, stream)) =
                    (&self.tensors[compute_target_id].data, &self.device)
                {
                    let f_fill = self.functions.get("fill_f32").unwrap().clone();
                    let n = 1u64;
                    let val = 0.0f32;
                    let mut builder = stream.launch_builder(&f_fill);
                    builder.arg(s).arg(&val).arg(&n);
                    unsafe { builder.launch(LaunchConfig::for_num_elems(1)) }.unwrap();
                }

                let l_s = match (&self.tensors[logits_id].data, &l_temp_fwd) {
                    (Storage::Gpu(s), _) => s,
                    #[cfg(feature = "bf16")] (_, Some(t)) => t,
                    _ => unreachable!("cross_entropy: logits must be Gpu or GpuBf16"),
                };
                let o_s = match &self.tensors[compute_target_id].data {
                    Storage::Gpu(s) => s,
                    _ => unreachable!(),
                };

                let b_u64 = batch_size as u64;
                let c_u64 = num_classes as u64;

                let mut builder = stream.launch_builder(&f_fwd);
                builder
                    .arg(l_s)
                    .arg(&targets_d)
                    .arg(o_s)
                    .arg(&b_u64)
                    .arg(&c_u64);
                unsafe { builder.launch(LaunchConfig::for_num_elems(batch_size as u32)) }.unwrap();

                let backward_fn = Box::new(move |tensors: &mut [Tensor]| {
                    #[cfg(feature = "bf16")]
                    let l_temp_bwd = crate::bf16_util::bf16_to_f32_temp(&tensors[logits_id].data, batch_size * num_classes, &stream_clone, &f_cast_to_f32_bwd);
                    #[cfg(not(feature = "bf16"))]
                    let l_temp_bwd: Option<()> = None;

                    let l_data = match (&tensors[logits_id].data, &l_temp_bwd) {
                        (Storage::Gpu(s), _) => s,
                        #[cfg(feature = "bf16")] (_, Some(t)) => t,
                        _ => unreachable!("cross_entropy backward: logits must be Gpu or GpuBf16"),
                    };

                    let out_grad = match &tensors[out_id].grad {
                        Storage::Gpu(s) => s,
                        _ => unreachable!("cross_entropy backward: out_grad must be Gpu"),
                    };
                    let l_grad = match &tensors[logits_id].grad {
                        Storage::Gpu(s) => s,
                        s => unreachable!("cross_entropy backward: l_grad must be Gpu [{:?}]", s),
                    };

                    let mut b1 = stream_clone.launch_builder(&f_bwd);
                    b1.arg(l_data)
                        .arg(&targets_d)
                        .arg(out_grad)
                        .arg(l_grad)
                        .arg(&b_u64)
                        .arg(&c_u64);
                    unsafe { b1.launch(LaunchConfig::for_num_elems(batch_size as u32)) }.unwrap();
                });

                #[cfg(feature = "bf16")]
                if cast_to_bf16_after {
                    let f_cast = self.functions.get("cast_f32_to_bf16").unwrap().clone();
                    match (&self.tensors[compute_target_id].data, &self.tensors[out_id].data) {
                        (Storage::Gpu(f32_src), Storage::GpuBf16(bf16_dst)) => {
                            let mut b = stream.launch_builder(&f_cast);
                            let n_elem = 1u64;
                            b.arg(f32_src).arg(bf16_dst).arg(&n_elem);
                            unsafe { b.launch(LaunchConfig::for_num_elems(1)) }.unwrap();
                            stream.synchronize().unwrap();
                        }
                        _ => {}
                    }
                    self.tensors.pop();
                }

                if !self.no_grad {
                    self.tape.nodes.push(TapeNode {
                        inputs: vec![logits_id],
                        output: out_id,
                        backward_fn,
                    });
                }
                out_id
            }
            Device::Cpu => {
                let logits = &self.tensors[logits_id];
                let batch_size = logits.shape[0];
                let num_classes = logits.shape[1];
                let mut out_loss = 0.0;
                let mut probs = vec![0.0; batch_size * num_classes];
                let logits_data = logits.data.as_cpu();

                for b in 0..batch_size {
                    let mut max_val = f32::NEG_INFINITY;
                    for c in 0..num_classes {
                        max_val = max_val.max(logits_data[b * num_classes + c]);
                    }
                    let mut sum_exp = 0.0;
                    for c in 0..num_classes {
                        let exp_val = (logits_data[b * num_classes + c] - max_val).exp();
                        probs[b * num_classes + c] = exp_val;
                        sum_exp += exp_val;
                    }
                    for c in 0..num_classes {
                        probs[b * num_classes + c] /= sum_exp;
                    }
                    out_loss += -(probs[b * num_classes + targets[b]] + 1e-8).ln();
                }

                let out_id = self.alloc(vec![], vec![out_loss / batch_size as f32]);
                let targets_cap = targets.to_vec();
                let backward_fn = Box::new(move |tensors: &mut [Tensor]| {
                    let o_grad = tensors[out_id].grad.as_cpu()[0];
                    let l_grad = tensors[logits_id].grad.as_cpu_mut();
                    for b in 0..batch_size {
                        for c in 0..num_classes {
                            let idx = b * num_classes + c;
                            let mut g = probs[idx];
                            if c == targets_cap[b] {
                                g -= 1.0;
                            }
                            l_grad[idx] += (g / batch_size as f32) * o_grad;
                        }
                    }
                });
                if !self.no_grad {
                    self.tape.nodes.push(TapeNode {
                        inputs: vec![logits_id],
                        output: out_id,
                        backward_fn,
                    });
                }
                out_id
            }
        }
    }

    /// Cross-entropy loss where targets equal to `IGNORE_INDEX` (usize::MAX)
    /// are silently skipped.  Falls back to `cross_entropy` if every target
    /// is valid (no masked positions) to avoid the normaliser overhead.
    pub fn cross_entropy_masked(&mut self, logits_id: usize, targets: &[usize]) -> usize {
        let valid_count = targets.iter().filter(|&&t| t != IGNORE_INDEX).count();
 
        // No masked positions → use standard (faster) CE.
        if valid_count == targets.len() {
            return self.cross_entropy(logits_id, targets);
        }
 
        // All positions masked — return zero loss (should not happen in practice).
        if valid_count == 0 {
            return self.alloc(vec![], vec![0.0]);
        }
 
        let normalizer = 1.0_f32 / valid_count as f32;
 
        let device = self.device.clone();
        match &device {
            Device::Gpu(_, stream) => {
                let logits      = &self.tensors[logits_id];
                let batch_size  = logits.shape[0];
                let num_classes = logits.shape[1];
 
                let out_id  = self.alloc_pooled(vec![]);
                self.name_tensor(out_id, "cross_entropy_masked_output");
                #[cfg(feature = "bf16")]
                let (compute_target_id, cast_to_bf16_after) = if self.uses_bf16_mixed_precision() {
                    let f32_slice = self.safe_alloc_zeros::<f32>(stream, 1);
                    let tmp_id = self.tensors.len();
                    let grad_slice = self.safe_alloc_zeros::<f32>(stream, 1);
                    self.tensors.push(Tensor {
                        id: tmp_id, shape: vec![],
                        strides: vec![],
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
                let f_fwd   = self.functions.get("cross_entropy_masked_f32").unwrap().clone();
                let f_bwd   = self.functions.get("cross_entropy_masked_backward_f32").unwrap().clone();
                let stream_clone = stream.clone();

                if let (Storage::Gpu(s), Device::Gpu(_, stream)) =
                    (&self.tensors[compute_target_id].data, &self.device)
                {
                    let f_fill = self.functions.get("fill_f32").unwrap().clone();
                    let n = 1u64;
                    let val = 0.0f32;
                    let mut builder = stream.launch_builder(&f_fill);
                    builder.arg(s).arg(&val).arg(&n);
                    unsafe { builder.launch(LaunchConfig::for_num_elems(1)) }.unwrap();
                }
 
                let targets_f32: Vec<f32> = targets
                    .iter()
                    .map(|&t| if t == IGNORE_INDEX { u32::MAX as f32 } else { t as f32 })
                    .collect();
                let targets_d = stream.clone_htod(targets_f32.as_slice()).unwrap();
 
                // Optional BF16 cast for logits (same pattern as cross_entropy).
                #[cfg(feature = "bf16")]
                let f_cast_to_f32 = self.functions.get("cast_bf16_to_f32").unwrap().clone();
                #[cfg(feature = "bf16")]
                let f_cast_to_f32_bwd = f_cast_to_f32.clone();
 
                #[cfg(feature = "bf16")]
                let l_temp_fwd = safe_bf16_temp!(
                    self, logits_id, batch_size * num_classes, stream, &f_cast_to_f32
                );
                #[cfg(not(feature = "bf16"))]
                let l_temp_fwd: Option<()> = None;
 
                let l_s = match (&self.tensors[logits_id].data, &l_temp_fwd) {
                    (Storage::Gpu(s), _) => s,
                    #[cfg(feature = "bf16")]
                    (_, Some(t)) => t,
                    _ => unreachable!("cross_entropy_masked: logits must be Gpu or GpuBf16"),
                };
                let o_s = match &self.tensors[compute_target_id].data {
                    Storage::Gpu(s) => s,
                    _ => unreachable!(),
                };
 
                let b_u64 = batch_size  as u64;
                let c_u64 = num_classes as u64;
 
                let mut builder = stream.launch_builder(&f_fwd);
                builder
                    .arg(l_s)
                    .arg(&targets_d)
                    .arg(o_s)
                    .arg(&normalizer)
                    .arg(&b_u64)
                    .arg(&c_u64);
                unsafe { builder.launch(LaunchConfig::for_num_elems(batch_size as u32)) }.unwrap();
 
                let backward_fn = Box::new(move |tensors: &mut [Tensor]| {
                    #[cfg(feature = "bf16")]
                    let l_temp_bwd = crate::bf16_util::bf16_to_f32_temp(
                        &tensors[logits_id].data,
                        batch_size * num_classes,
                        &stream_clone,
                        &f_cast_to_f32_bwd,
                    );
                    #[cfg(not(feature = "bf16"))]
                    let l_temp_bwd: Option<()> = None;
 
                    let l_data = match (&tensors[logits_id].data, &l_temp_bwd) {
                        (Storage::Gpu(s), _) => s,
                        #[cfg(feature = "bf16")]
                        (_, Some(t)) => t,
                        _ => unreachable!(),
                    };
                    let out_grad = match &tensors[out_id].grad {
                        Storage::Gpu(s) => s,
                        _ => unreachable!(),
                    };
                    let l_grad = match &tensors[logits_id].grad {
                        Storage::Gpu(s) => s,
                        _ => unreachable!(),
                    };
 
                    let mut b1 = stream_clone.launch_builder(&f_bwd);
                    b1.arg(l_data)
                        .arg(&targets_d)
                        .arg(out_grad)
                        .arg(l_grad)
                        .arg(&normalizer)
                        .arg(&b_u64)
                        .arg(&c_u64);
                    unsafe { b1.launch(LaunchConfig::for_num_elems(batch_size as u32)) }.unwrap();
                });

                #[cfg(feature = "bf16")]
                if cast_to_bf16_after {
                    let f_cast = self.functions.get("cast_f32_to_bf16").unwrap().clone();
                    match (&self.tensors[compute_target_id].data, &self.tensors[out_id].data) {
                        (Storage::Gpu(f32_src), Storage::GpuBf16(bf16_dst)) => {
                            let mut b = stream.launch_builder(&f_cast);
                            let n_elem = 1u64;
                            b.arg(f32_src).arg(bf16_dst).arg(&n_elem);
                            unsafe { b.launch(LaunchConfig::for_num_elems(1)) }.unwrap();
                            stream.synchronize().unwrap();
                        }
                        _ => {}
                    }
                    self.tensors.pop();
                }
 
                if !self.no_grad {
                    self.tape.nodes.push(TapeNode {
                        inputs: vec![logits_id],
                        output: out_id,
                        backward_fn,
                    });
                }
                out_id
            }
 
            Device::Cpu => {
                // CPU reference path (used by tests / small models).
                let logits      = &self.tensors[logits_id];
                let batch_size  = logits.shape[0];
                let num_classes = logits.shape[1];
                let logits_data = logits.data.as_cpu().clone();
 
                let mut out_loss = 0.0_f32;
                let mut probs    = vec![0.0_f32; batch_size * num_classes];
 
                for b in 0..batch_size {
                    if targets[b] == IGNORE_INDEX { continue; }
                    let mut max_val = f32::NEG_INFINITY;
                    for c in 0..num_classes {
                        max_val = max_val.max(logits_data[b * num_classes + c]);
                    }
                    let mut sum_exp = 0.0_f32;
                    for c in 0..num_classes {
                        let e = (logits_data[b * num_classes + c] - max_val).exp();
                        probs[b * num_classes + c] = e;
                        sum_exp += e;
                    }
                    for c in 0..num_classes {
                        probs[b * num_classes + c] /= sum_exp;
                    }
                    let p = probs[b * num_classes + targets[b]];
                    out_loss += -(p + 1e-8).ln() * normalizer;
                }
 
                let targets_cap = targets.to_vec();
                let out_id = self.alloc(vec![], vec![out_loss]);
 
                let backward_fn = Box::new(move |tensors: &mut [Tensor]| {
                    let o_grad  = tensors[out_id].grad.as_cpu()[0];
                    let l_grad  = tensors[logits_id].grad.as_cpu_mut();
                    for b in 0..batch_size {
                        if targets_cap[b] == IGNORE_INDEX { continue; }
                        for c in 0..num_classes {
                            let idx = b * num_classes + c;
                            let mut g = probs[idx];
                            if c == targets_cap[b] { g -= 1.0; }
                            l_grad[idx] += g * normalizer * o_grad;
                        }
                    }
                });
 
                if !self.no_grad {
                    self.tape.nodes.push(TapeNode {
                        inputs: vec![logits_id],
                        output: out_id,
                        backward_fn,
                    });
                }
                out_id
            }
        }
    }
}
