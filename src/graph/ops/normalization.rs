use crate::graph::{Graph, TapeNode, is_bf16};
use crate::tensor::{Tensor, Device, Storage};
use cudarc::driver::{PushKernelArg, LaunchConfig, CudaSlice};


impl Graph {

    pub fn rms_norm(&mut self, x_id: usize, weight_id: usize, eps: f32) -> usize {
        let device = self.device.clone();
        match &device {
            Device::Gpu(_, stream) => {
                let x = &self.tensors[x_id];
                let dim = *x.shape.last().unwrap();
                let num_vecs = x.data.len() / dim;
                let stream_clone = stream.clone();

                #[cfg(feature = "bf16")]
                let (f_fwd, f_bwd) = {
                    let x_bf = is_bf16(&self.tensors[x_id].data);
                    let w_bf = is_bf16(&self.tensors[weight_id].data);
                    match (x_bf, w_bf) {
                        (true, true) => (
                            self.functions.get("rmsnorm_bf16").unwrap().clone(),
                            self.functions.get("rmsnorm_backward_bf16in_f32").unwrap().clone(),
                        ),
                        (false, true) => (
                            self.functions.get("rmsnorm_f32_bf16w").unwrap().clone(),
                            self.functions.get("rmsnorm_backward_bf16w_f32").unwrap().clone(),
                        ),
                        _ => (
                            self.functions.get("rmsnorm_f32").unwrap().clone(),
                            self.functions.get("rmsnorm_backward_f32").unwrap().clone(),
                        ),
                    }
                };
                #[cfg(not(feature = "bf16"))]
                let (f_fwd, f_bwd) = (
                    self.functions.get("rmsnorm_f32").unwrap().clone(),
                    self.functions.get("rmsnorm_backward_f32").unwrap().clone(),
                );

                let b_u16_slice: CudaSlice<u16>;
                let out_id = self.alloc_pooled(x.shape.clone());
                self.name_tensor(out_id, "rmsnorm_output");
                let (dim_u64, num_vecs_u64) = (dim as u64, num_vecs as u64);

                self.ensure_on_gpu(x_id);
                self.ensure_on_gpu(weight_id);
                self.ensure_on_gpu(out_id);

                {
                    let mut builder = stream.launch_builder(&f_fwd);
                    match (
                        &self.tensors[x_id].data,
                        &self.tensors[weight_id].data,
                        &self.tensors[out_id].data,
                    ) {
                        (Storage::Gpu(x_s), Storage::Gpu(w_s), Storage::Gpu(o_s)) => {
                            builder.arg(x_s).arg(w_s).arg(o_s)
                                .arg(&dim_u64).arg(&eps).arg(&num_vecs_u64);
                        }
                        #[cfg(feature = "bf16")]
                        (Storage::GpuBf16(x_s), Storage::Gpu(w_s), Storage::Gpu(o_s)) => {
                            builder.arg(x_s).arg(w_s).arg(o_s)
                                .arg(&dim_u64).arg(&eps).arg(&num_vecs_u64);
                        }
                        #[cfg(feature = "bf16")]
                        (Storage::Gpu(x_s), Storage::GpuBf16(w_s), Storage::Gpu(o_s)) => {
                            builder.arg(x_s).arg(w_s).arg(o_s)
                                .arg(&dim_u64).arg(&eps).arg(&num_vecs_u64);
                        }
                        #[cfg(feature = "bf16")]
                        (Storage::GpuBf16(x_s), Storage::GpuBf16(w_s), Storage::GpuBf16(o_s)) => {
                            builder.arg(x_s).arg(w_s).arg(o_s)
                                .arg(&dim_u64).arg(&eps).arg(&num_vecs_u64);
                        }
                        #[cfg(feature = "bf16")]
                        (Storage::GpuBf16(x_s), Storage::Cpu(w_s), Storage::GpuBf16(o_s)) => {
                            let stream = match &self.device { Device::Gpu(_, s) => s.clone(), _ => unreachable!() };
                            b_u16_slice = stream
                                .clone_htod(&w_s.as_slice().iter().map(|&x| x as u16).collect::<Vec<u16>>())
                                .expect("bf16_streamed: forward rms_norm failed");

                            builder.arg(x_s).arg(&b_u16_slice).arg(o_s)
                                .arg(&dim_u64).arg(&eps).arg(&num_vecs_u64);
                        }
                        (p1, p2, p3) => panic!(
                            "rms_norm: unsupported storage combination. Received: ({:?}, {:?}, {:?})",
                            p1, p2, p3
                        ),
                    }
                    unsafe { builder.launch(LaunchConfig::for_num_elems(num_vecs as u32)) }.unwrap();
                }

                let backward_fn = Box::new(move |tensors: &mut [Tensor]| {
                    let out_grad = match &tensors[out_id].grad {
                        Storage::Gpu(s) => s, _ => unreachable!()
                    };
                    let x_grad = match &tensors[x_id].grad {
                        Storage::Gpu(s) => s, _ => unreachable!()
                    };
                    let w_grad = match &tensors[weight_id].grad {
                        Storage::Gpu(s) => s, _ => unreachable!()
                    };
                    let mut builder = stream_clone.launch_builder(&f_bwd);
                    match (&tensors[x_id].data, &tensors[weight_id].data) {
                        (Storage::Gpu(x_s), Storage::Gpu(w_s)) => {
                            builder.arg(x_s).arg(w_s).arg(out_grad)
                                .arg(x_grad).arg(w_grad)
                                .arg(&dim_u64).arg(&eps).arg(&num_vecs_u64);
                        }
                        #[cfg(feature = "bf16")]
                        (Storage::Gpu(x_s), Storage::GpuBf16(w_s)) => {
                            builder.arg(x_s).arg(w_s).arg(out_grad)
                                .arg(x_grad).arg(w_grad)
                                .arg(&dim_u64).arg(&eps).arg(&num_vecs_u64);
                        }
                        #[cfg(feature = "bf16")]
                        (Storage::GpuBf16(x_s), Storage::GpuBf16(w_s)) => {
                            builder.arg(x_s).arg(w_s).arg(out_grad)
                                .arg(x_grad).arg(w_grad)
                                .arg(&dim_u64).arg(&eps).arg(&num_vecs_u64);
                        }
                        _ => panic!("rms_norm backward: unsupported storage"),
                    }
                    unsafe { builder.launch(LaunchConfig::for_num_elems(num_vecs as u32)) }.unwrap();
                });

                if !self.no_grad {
                    self.tape.nodes.push(TapeNode {
                        inputs: vec![x_id, weight_id], output: out_id, backward_fn,
                    });
                }
                out_id
            }
            Device::Cpu => {
                let x = &self.tensors[x_id];
                let dim = *x.shape.last().unwrap();
                let num_vecs = x.data.as_cpu().len() / dim;
                let mut out_data = vec![0.0; x.data.as_cpu().len()];
                let mut rrms_cache = vec![0.0; num_vecs];
                let x_fwd = x.data.as_cpu().clone();
                let w_fwd = self.tensors[weight_id].data.as_cpu().clone();

                for n in 0..num_vecs {
                    let off = n * dim;
                    let mut ss = 0.0;
                    for d in 0..dim {
                        ss += x_fwd[off + d].powi(2);
                    }
                    let rrms = 1.0 / (ss / dim as f32 + eps).sqrt();
                    rrms_cache[n] = rrms;
                    for d in 0..dim {
                        out_data[off + d] = x_fwd[off + d] * rrms * w_fwd[d];
                    }
                }

                let out_id = self.alloc(x.shape.clone(), out_data);
                let backward_fn = Box::new(move |tensors: &mut [Tensor]| {
                    let o_grad = tensors[out_id].grad.as_cpu().clone();
                    for n in 0..num_vecs {
                        let off = n * dim;
                        let rrms = rrms_cache[n];
                        let mut gdxw = 0.0;
                        for d in 0..dim {
                            gdxw += o_grad[off + d] * x_fwd[off + d] * w_fwd[d];
                        }
                        let rrc_d = (rrms.powi(3)) / dim as f32;
                        for d in 0..dim {
                            let dx =
                                rrms * (o_grad[off + d] * w_fwd[d]) - x_fwd[off + d] * rrc_d * gdxw;
                            tensors[x_id].grad.as_cpu_mut()[off + d] += dx;
                            tensors[weight_id].grad.as_cpu_mut()[d] +=
                                o_grad[off + d] * x_fwd[off + d] * rrms;
                        }
                    }
                });
                if !self.no_grad {
                    self.tape.nodes.push(TapeNode {
                        inputs: vec![x_id, weight_id],
                        output: out_id,
                        backward_fn,
                    });
                }
                out_id
            }
        }
    }

}
