use crate::graph::Graph;
use crate::tape::TapeNode;
use crate::tensor::{Storage, Tensor, Device};
use cudarc::driver::{PushKernelArg, LaunchConfig};

#[cfg(feature = "bf16")]
use crate::bf16_util::is_bf16;

impl Graph {
    /// Repeats the Key or Value heads to match the number of Query heads for GQA.
    /// Expects input shape: [Batch, NumKVHeads, SeqLen, HeadDim]
    /// Returns shape:       [Batch, NumKVHeads * repeats, SeqLen, HeadDim]
    pub fn repeat_kv(&mut self, x_id: usize, repeats: usize) -> usize {
        let in_shape = self.tensors[x_id].shape.clone();
        assert_eq!(
            in_shape.len(),
            4,
            "repeat_kv requires [Batch, KVHeads, SeqLen, HeadDim]"
        );

        if repeats == 1 {
            return x_id; // No-op for standard Multi-Head Attention
        }

        let batch = in_shape[0];
        let num_kv_heads = in_shape[1];
        let seq_len = in_shape[2];
        let head_dim = in_shape[3];

        let num_q_heads = num_kv_heads * repeats;
        let out_shape = vec![batch, num_q_heads, seq_len, head_dim];
        let total_out_elements = out_shape.iter().product::<usize>();
        let total_kv_elements = in_shape.iter().product::<usize>();

        let device = self.device.clone();

        match &device {
            Device::Gpu(_, stream) => {
                let stream_clone = stream.clone();

                #[cfg(feature = "bf16")]
                let (f_fwd, f_bwd) = if is_bf16(&self.tensors[x_id].data) {
                    (
                        self.functions.get("repeat_kv_bf16").unwrap().clone(),
                        self.functions.get("repeat_kv_backward_bf16").unwrap().clone(),
                    )
                } else {
                    (
                        self.functions.get("repeat_kv_f32").unwrap().clone(),
                        self.functions.get("repeat_kv_backward_f32").unwrap().clone(),
                    )
                };

                #[cfg(not(feature = "bf16"))]
                let (f_fwd, f_bwd) = (
                    self.functions.get("repeat_kv_f32").unwrap().clone(),
                    self.functions.get("repeat_kv_backward_f32").unwrap().clone(),
                );

                let out_id = self.alloc_pooled(out_shape);
                self.name_tensor(out_id, "gqa_output");

                // Convert dimensions to i32 for the CUDA kernel arguments
                let (b_i32, kv_h_i32, rep_i32, s_i32, d_i32) = (
                    batch as i32,
                    num_kv_heads as i32,
                    repeats as i32,
                    seq_len as i32,
                    head_dim as i32,
                );

                self.ensure_on_gpu(x_id);
                self.ensure_on_gpu(out_id);

                {
                    let mut builder = stream.launch_builder(&f_fwd);
                    match (&self.tensors[x_id].data, &self.tensors[out_id].data) {
                        (Storage::Gpu(in_s), Storage::Gpu(out_s)) => {
                            builder
                                .arg(in_s)
                                .arg(out_s)
                                .arg(&b_i32)
                                .arg(&kv_h_i32)
                                .arg(&rep_i32)
                                .arg(&s_i32)
                                .arg(&d_i32);
                        }
                        #[cfg(feature = "bf16")]
                        (Storage::GpuBf16(in_s), Storage::GpuBf16(out_s)) => {
                            builder
                                .arg(in_s)
                                .arg(out_s)
                                .arg(&b_i32)
                                .arg(&kv_h_i32)
                                .arg(&rep_i32)
                                .arg(&s_i32)
                                .arg(&d_i32);
                        }
                        _ => panic!("repeat_kv: mismatched storage"),
                    }
                    unsafe {
                        builder.launch(LaunchConfig::for_num_elems(total_out_elements as u32))
                    }
                    .unwrap();
                }

                let backward_fn = Box::new(move |tensors: &mut [Tensor]| {
                    let mut b1 = stream_clone.launch_builder(&f_bwd);

                    match (&tensors[out_id].grad, &tensors[x_id].grad) {
                        (Storage::Gpu(out_grad), Storage::Gpu(in_grad)) => {
                            b1.arg(out_grad)
                                .arg(in_grad)
                                .arg(&b_i32)
                                .arg(&kv_h_i32)
                                .arg(&rep_i32)
                                .arg(&s_i32)
                                .arg(&d_i32);
                        }
                        #[cfg(feature = "bf16")]
                        (Storage::GpuBf16(out_grad), Storage::GpuBf16(in_grad)) => {
                            b1.arg(out_grad)
                                .arg(in_grad)
                                .arg(&b_i32)
                                .arg(&kv_h_i32)
                                .arg(&rep_i32)
                                .arg(&s_i32)
                                .arg(&d_i32);
                        }
                        _ => panic!("repeat_kv_bw: mismatched or missing gradient storage"),
                    }

                    unsafe { b1.launch(LaunchConfig::for_num_elems(total_kv_elements as u32)) }
                        .unwrap();
                });

                if !self.no_grad {
                    self.tape.nodes.push(TapeNode {
                        inputs: vec![x_id],
                        output: out_id,
                        backward_fn,
                    });
                }
                out_id
            }
            Device::Cpu => {
                let in_data = self.tensors[x_id].data.as_cpu().clone();
                let mut out_data = vec![0.0; total_out_elements];

                for b in 0..batch {
                    for q_h in 0..num_q_heads {
                        let kv_h = q_h / repeats;
                        for s in 0..seq_len {
                            for d in 0..head_dim {
                                let in_idx =
                                    ((b * num_kv_heads + kv_h) * seq_len + s) * head_dim + d;
                                let out_idx =
                                    ((b * num_q_heads + q_h) * seq_len + s) * head_dim + d;
                                out_data[out_idx] = in_data[in_idx];
                            }
                        }
                    }
                }

                let out_id = self.alloc(out_shape, out_data);

                let backward_fn = Box::new(move |tensors: &mut [Tensor]| {
                    let out_grad = tensors[out_id].grad.as_cpu().clone();
                    let in_grad = tensors[x_id].grad.as_cpu_mut();

                    for b in 0..batch {
                        for kv_h in 0..num_kv_heads {
                            for r in 0..repeats {
                                let q_h = kv_h * repeats + r;
                                for s in 0..seq_len {
                                    for d in 0..head_dim {
                                        let in_idx =
                                            ((b * num_kv_heads + kv_h) * seq_len + s) * head_dim + d;
                                        let out_idx =
                                            ((b * num_q_heads + q_h) * seq_len + s) * head_dim + d;
                                        in_grad[in_idx] += out_grad[out_idx];
                                    }
                                }
                            }
                        }
                    }
                });

                if !self.no_grad {
                    self.tape.nodes.push(TapeNode {
                        inputs: vec![x_id],
                        output: out_id,
                        backward_fn,
                    });
                }
                out_id
            }
        }
    }
}
