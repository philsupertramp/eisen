use crate::graph::{Graph, is_bf16};
use crate::tensor::{Tensor, Device, Storage};
use cudarc::driver::{PushKernelArg, LaunchConfig};
use crate::safe_bf16_temp;

impl Graph {
    pub fn flash_attention(
        &mut self,
        q_id: usize,
        k_id: usize,
        v_id: usize,
        scale: f32,
        causal: bool,
    ) -> usize {
        assert!(
            self.no_grad,
            "flash_attention currently supports no_grad/inference path only"
        );
        let q_shape = self.tensors[q_id].shape.clone();
        let k_shape = self.tensors[k_id].shape.clone();
        let v_shape = self.tensors[v_id].shape.clone();
        assert_eq!(q_shape.len(), 3, "flash_attention expects q=[B,M,D]");
        assert_eq!(k_shape.len(), 3, "flash_attention expects k=[B,N,D]");
        assert_eq!(v_shape.len(), 3, "flash_attention expects v=[B,N,D]");

        let batch = q_shape[0];
        let m = q_shape[1];
        let d = q_shape[2];
        let n = k_shape[1];

        assert_eq!(k_shape[0], batch, "flash_attention batch mismatch q/k");
        assert_eq!(v_shape[0], batch, "flash_attention batch mismatch q/v");
        assert_eq!(k_shape[2], d, "flash_attention head dim mismatch q/k");
        assert_eq!(v_shape[1], n, "flash_attention sequence mismatch k/v");
        assert_eq!(v_shape[2], d, "flash_attention head dim mismatch q/v");
        assert!(
            d <= 256,
            "flash_attention kernel currently supports head_dim <= 256"
        );

        let device = self.device.clone();
        match &device {
            Device::Gpu(_, stream) => {
                let f_fwd = self.functions.get("flash_attention_f32").unwrap().clone();
                let out_id = self.alloc_pooled(vec![batch, m, d]);
                self.name_tensor(out_id, "flash_attention_output");

                #[cfg(feature = "bf16")]
                let (compute_target_id, cast_to_bf16_after) = if self.uses_bf16_mixed_precision() {
                    let f32_slice = self.safe_alloc_zeros::<f32>(&stream, batch * m * d);
                    let tmp_id = self.tensors.len();
                    let grad_slice = self.safe_alloc_zeros::<f32>(&stream, 1);
                    self.tensors.push(Tensor {
                        id: tmp_id, shape: vec![batch, m, d],
                        strides: Tensor::compute_strides(&[batch, m, d]),
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
                let q_temp_fwd = safe_bf16_temp!(self, q_id, batch * m * d, stream, &f_cast_to_f32);
                #[cfg(not(feature = "bf16"))]
                let q_temp_fwd: Option<()> = None;

                #[cfg(feature = "bf16")]
                let k_temp_fwd = safe_bf16_temp!(self, k_id, batch * n * d, stream, &f_cast_to_f32);
                #[cfg(not(feature = "bf16"))]
                let k_temp_fwd: Option<()> = None;

                #[cfg(feature = "bf16")]
                let v_temp_fwd = safe_bf16_temp!(self, v_id, batch * n * d, stream, &f_cast_to_f32);
                #[cfg(not(feature = "bf16"))]
                let v_temp_fwd: Option<()> = None;


                let q_s = match (&self.tensors[q_id].data, &q_temp_fwd) {
                    (Storage::Gpu(s), _) => s,
                    #[cfg(feature = "bf16")] (_, Some(t)) => t,
                    _ => unreachable!(),
                };
                let k_s = match (&self.tensors[k_id].data, &k_temp_fwd) {
                    (Storage::Gpu(s), _) => s,
                    #[cfg(feature = "bf16")] (_, Some(t)) => t,
                    _ => unreachable!(),
                };
                let v_s = match (&self.tensors[v_id].data, &v_temp_fwd) {
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
                let n_u64 = n as u64;
                let d_u64 = d as u64;
                let total_rows = (batch * m) as u32;

                let mut builder = stream.launch_builder(&f_fwd);
                builder
                    .arg(q_s)
                    .arg(k_s)
                    .arg(v_s)
                    .arg(o_s)
                    .arg(&batch_u64)
                    .arg(&m_u64)
                    .arg(&n_u64)
                    .arg(&d_u64)
                    .arg(&scale)
                    .arg(&causal);
                unsafe { builder.launch(LaunchConfig::for_num_elems(total_rows)) }.unwrap();
                
                #[cfg(feature = "bf16")]
                if cast_to_bf16_after {
                    let f_cast = self.functions.get("cast_f32_to_bf16").unwrap().clone();
                    let n_elem = (batch * m * d) as u64;
                    match (&self.tensors[compute_target_id].data, &self.tensors[out_id].data) {
                        (Storage::Gpu(f32_src), Storage::GpuBf16(bf16_dst)) => {
                            let mut b = stream.launch_builder(&f_cast);
                            b.arg(f32_src).arg(bf16_dst).arg(&n_elem);
                            unsafe { b.launch(LaunchConfig::for_num_elems((batch * m * d) as u32)) }.unwrap();
                            stream.synchronize().unwrap();
                        }
                        _ => {}
                    }
                    self.tensors.pop();
                }

                out_id
            }
            Device::Cpu => {
                let q_data = self.tensors[q_id].data.as_cpu();
                let k_data = self.tensors[k_id].data.as_cpu();
                let v_data = self.tensors[v_id].data.as_cpu();
                let mut out = vec![0.0; batch * m * d];
                let mut scores = vec![0.0; n];
                let mut probs = vec![0.0; n];

                for bb in 0..batch {
                    for i in 0..m {
                        for j in 0..n {
                            if causal && j > i {
                                scores[j] = f32::NEG_INFINITY;
                                continue;
                            }
                            let mut dot = 0.0;
                            for dd in 0..d {
                                let q_idx = ((bb * m + i) * d) + dd;
                                let k_idx = ((bb * n + j) * d) + dd;
                                dot += q_data[q_idx] * k_data[k_idx];
                            }
                            scores[j] = dot * scale;
                        }

                        let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                        let mut denom = 0.0;
                        for j in 0..n {
                            let e = (scores[j] - max_score).exp();
                            probs[j] = e;
                            denom += e;
                        }
                        for j in 0..n {
                            probs[j] /= denom;
                        }

                        for dd in 0..d {
                            let mut acc = 0.0;
                            for j in 0..n {
                                let v_idx = ((bb * n + j) * d) + dd;
                                acc += probs[j] * v_data[v_idx];
                            }
                            out[((bb * m + i) * d) + dd] = acc;
                        }
                    }
                }

                self.alloc(vec![batch, m, d], out)
            }
        }
    }
}
