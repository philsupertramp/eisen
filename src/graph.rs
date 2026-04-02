use crate::tape::{Tape, TapeNode};
use crate::tensor::{Tensor, Device, Storage};
use cudarc::driver::{LaunchConfig, CudaFunction, PushKernelArg};
use std::collections::HashMap;

pub struct Graph {
    pub tensors: Vec<Tensor>,
    pub tape: Tape,
    pub device: Device,
    /// Store loaded CUDA functions to avoid re-loading handles every op
    pub functions: HashMap<String, CudaFunction>,
}

impl Default for Graph {
    fn default() -> Self {
        Self::new(Device::Cpu)
    }
}

impl Graph {
    pub fn new(device: Device) -> Self {
        let mut functions = HashMap::new();

        // If we are on GPU, load our pre-compiled kernels
        if let Device::Gpu(ctx, _) = &device {
            let ptx = include_str!(concat!(env!("OUT_DIR"), "/ops.ptx"));
            let module = ctx
                .load_module(ptx.into())
                .expect("Failed to load PTX module");

            let names = [
                "add_f32", "fill_f32", "accumulate_f32", 
                "mul_f32", "mul_backward_f32",
                "matmul_f32", "matmul_backward_a_f32", "matmul_backward_b_f32",
                "silu_f32", "silu_backward_f32",
                "gather_f32", "gather_backward_f32",
                "rmsnorm_f32", "rmsnorm_backward_f32",
                "copy_f32", "cross_entropy_f32", "cross_entropy_backward_f32",
                "sum_f32", "sum_backward_f32", "max_f32", "max_backward_f32",
            ];
            for name in names {
                let f = module.load_function(name).expect(&format!("Failed to load {} kernel", name));
                functions.insert(name.to_string(), f);
            }
        }

        Self {
            tensors: Vec::new(),
            tape: Tape::default(),
            device,
            functions,
        }
    }

    /// Helper for tests to quickly get a value from a tensor's data buffer
    pub fn get_data(&self, tensor_id: usize) -> &Vec<f32> {
        self.tensors[tensor_id].data.as_cpu()
    }

    /// Helper for tests to quickly get a value from a tensor's grad buffer
    pub fn get_grad(&self, tensor_id: usize) -> &Vec<f32> {
        self.tensors[tensor_id].grad.as_cpu()
    }

    /// Pulls gradient data from VRAM back to a CPU Vec<f32>.
    pub fn sync_grad_to_cpu(&self, tensor_id: usize) -> Vec<f32> {
        match &self.tensors[tensor_id].grad {
            Storage::Cpu(v) => v.clone(),
            Storage::Gpu(s) => {
                let (_, stream) = match &self.device {
                    Device::Gpu(_, s) => (None::<f32>, s),
                    _ => unreachable!(),
                };
                stream.clone_dtoh(s).expect("Failed to copy grad from VRAM to Host")
            }
        }
    }
    pub fn alloc(&mut self, shape: Vec<usize>, data: Vec<f32>) -> usize {
        let id = self.tensors.len();
        let tensor = Tensor::new(id, shape, data, self.device.clone());
        self.tensors.push(tensor);
        id
    }

    pub fn add(&mut self, a_id: usize, b_id: usize) -> usize {
        let a_shape = self.tensors[a_id].shape.clone();
        let b_shape = self.tensors[b_id].shape.clone();
        let out_shape = Tensor::get_broadcasted_shape(&a_shape, &b_shape);
        let out_size = out_shape.iter().product::<usize>();

        let device = self.device.clone();

        match &device {
            Device::Gpu(_, stream) => {
                let a_strides = Tensor::get_broadcasted_strides(&a_shape, &self.tensors[a_id].strides, &out_shape);
                let b_strides = Tensor::get_broadcasted_strides(&b_shape, &self.tensors[b_id].strides, &out_shape);
                
                let rank = out_shape.len() as u64;
                assert!(rank <= 3, "GPU Add currently supports up to Rank 3");

                let mut s = [1u64; 3];
                let mut a_str = [0u64; 3];
                let mut b_str = [0u64; 3];
                for i in 0..out_shape.len() {
                    s[i] = out_shape[i] as u64;
                    a_str[i] = a_strides[i] as u64;
                    b_str[i] = b_strides[i] as u64;
                }

                let f_forward = self.functions.get("add_f32").expect("add_f32 not loaded").clone();
                let f_accumulate = self.functions.get("accumulate_f32").expect("accumulate_f32 not loaded").clone();
                let stream_clone = stream.clone();

                let out_id = self.alloc(out_shape.clone(), vec![0.0; out_size]);
                
                let a_s = match &self.tensors[a_id].data { Storage::Gpu(s) => s, _ => unreachable!() };
                let b_s = match &self.tensors[b_id].data { Storage::Gpu(s) => s, _ => unreachable!() };
                let o_s = match &self.tensors[out_id].data { Storage::Gpu(s) => s, _ => unreachable!() };
                
                let n = out_size as u64;
                let cfg = LaunchConfig::for_num_elems(out_size as u32);
                let mut builder = stream.launch_builder(&f_forward);
                builder.arg(a_s).arg(b_s).arg(o_s).arg(&n).arg(&rank)
                       .arg(&s[0]).arg(&s[1]).arg(&s[2])
                       .arg(&a_str[0]).arg(&a_str[1]).arg(&a_str[2])
                       .arg(&b_str[0]).arg(&b_str[1]).arg(&b_str[2]);
                unsafe { builder.launch(cfg) }.expect("Failed to launch add_f32");

                let backward_fn = Box::new(move |tensors: &mut [Tensor]| {
                    let out_grad = match &tensors[out_id].grad { Storage::Gpu(s) => s, _ => unreachable!() };
                    let cfg_grad = LaunchConfig::for_num_elems(out_size as u32);

                    let a_grad = match &tensors[a_id].grad { Storage::Gpu(s) => s, _ => unreachable!() };
                    let mut b1 = stream_clone.launch_builder(&f_accumulate);
                    b1.arg(a_grad).arg(out_grad).arg(&n).arg(&rank)
                      .arg(&s[0]).arg(&s[1]).arg(&s[2])
                      .arg(&a_str[0]).arg(&a_str[1]).arg(&a_str[2]);
                    unsafe { b1.launch(cfg_grad) }.expect("Failed to accumulate grad into A");

                    let b_grad = match &tensors[b_id].grad { Storage::Gpu(s) => s, _ => unreachable!() };
                    let mut b2 = stream_clone.launch_builder(&f_accumulate);
                    b2.arg(b_grad).arg(out_grad).arg(&n).arg(&rank)
                      .arg(&s[0]).arg(&s[1]).arg(&s[2])
                      .arg(&b_str[0]).arg(&b_str[1]).arg(&b_str[2]);
                    unsafe { b2.launch(cfg_grad) }.expect("Failed to accumulate grad into B");
                });

                self.tape.nodes.push(TapeNode { inputs: vec![a_id, b_id], output: out_id, backward_fn });
                out_id
            }
            Device::Cpu => {
                let a_strides = Tensor::get_broadcasted_strides(&a_shape, &self.tensors[a_id].strides, &out_shape);
                let b_strides = Tensor::get_broadcasted_strides(&b_shape, &self.tensors[b_id].strides, &out_shape);
                let mut out_data = vec![0.0; out_size];
                {
                    let a_data = self.tensors[a_id].data.as_cpu();
                    let b_data = self.tensors[b_id].data.as_cpu();
                    for i in 0..out_size {
                        let nd = Tensor::flat_to_nd(i, &out_shape);
                        out_data[i] = a_data[Tensor::nd_to_flat(&nd, &a_strides)] + b_data[Tensor::nd_to_flat(&nd, &b_strides)];
                    }
                }
                let out_id = self.alloc(out_shape.clone(), out_data);
                
                let backward_fn = Box::new(move |tensors: &mut [Tensor]| {
                    let out_grad = tensors[out_id].grad.as_cpu().clone();
                    for i in 0..out_size {
                        let nd = Tensor::flat_to_nd(i, &out_shape);
                        tensors[a_id].grad.as_cpu_mut()[Tensor::nd_to_flat(&nd, &a_strides)] += out_grad[i];
                        tensors[b_id].grad.as_cpu_mut()[Tensor::nd_to_flat(&nd, &b_strides)] += out_grad[i];
                    }
                });
                self.tape.nodes.push(TapeNode { inputs: vec![a_id, b_id], output: out_id, backward_fn });
                out_id
            }
        }
    }

    pub fn mul(&mut self, a_id: usize, b_id: usize) -> usize {
        let a_shape = self.tensors[a_id].shape.clone();
        let b_shape = self.tensors[b_id].shape.clone();
        let out_shape = Tensor::get_broadcasted_shape(&a_shape, &b_shape);
        let out_size: usize = out_shape.iter().product();
        
        let device = self.device.clone();

        match &device {
            Device::Gpu(_, stream) => {
                assert_eq!(a_shape, b_shape, "GPU Mul requires matching shapes");
                
                let f_fwd = self.functions.get("mul_f32").expect("mul_f32 not loaded").clone();
                let f_bwd = self.functions.get("mul_backward_f32").expect("mul_backward_f32 not loaded").clone();
                let stream_clone = stream.clone();

                let out_id = self.alloc(out_shape.clone(), vec![0.0; out_size]);
                
                let a_s = match &self.tensors[a_id].data { Storage::Gpu(s) => s, _ => unreachable!() };
                let b_s = match &self.tensors[b_id].data { Storage::Gpu(s) => s, _ => unreachable!() };
                let o_s = match &self.tensors[out_id].data { Storage::Gpu(s) => s, _ => unreachable!() };
                
                let n = out_size as u64;
                let mut builder = stream.launch_builder(&f_fwd);
                builder.arg(a_s).arg(b_s).arg(o_s).arg(&n);
                unsafe { builder.launch(LaunchConfig::for_num_elems(out_size as u32)) }.expect("Failed to launch mul_f32");

                let backward_fn = Box::new(move |tensors: &mut [Tensor]| {
                    let a_data = match &tensors[a_id].data { Storage::Gpu(s) => s, _ => unreachable!() };
                    let b_data = match &tensors[b_id].data { Storage::Gpu(s) => s, _ => unreachable!() };
                    let out_grad = match &tensors[out_id].grad { Storage::Gpu(s) => s, _ => unreachable!() };
                    let a_grad = match &tensors[a_id].grad { Storage::Gpu(s) => s, _ => unreachable!() };
                    let b_grad = match &tensors[b_id].grad { Storage::Gpu(s) => s, _ => unreachable!() };

                    let mut b1 = stream_clone.launch_builder(&f_bwd);
                    b1.arg(a_data).arg(b_data).arg(out_grad).arg(a_grad).arg(b_grad).arg(&n);
                    unsafe { b1.launch(LaunchConfig::for_num_elems(out_size as u32)) }.expect("Failed mul_backward_f32");
                });

                self.tape.nodes.push(TapeNode { inputs: vec![a_id, b_id], output: out_id, backward_fn });
                out_id
            }
            Device::Cpu => {
                let a_strides = Tensor::get_broadcasted_strides(&a_shape, &self.tensors[a_id].strides, &out_shape);
                let b_strides = Tensor::get_broadcasted_strides(&b_shape, &self.tensors[b_id].strides, &out_shape);
                let mut out_data = vec![0.0; out_size];
                let a_fwd = self.tensors[a_id].data.as_cpu().clone();
                let b_fwd = self.tensors[b_id].data.as_cpu().clone();
                for i in 0..out_size {
                    let nd = Tensor::flat_to_nd(i, &out_shape);
                    out_data[i] = a_fwd[Tensor::nd_to_flat(&nd, &a_strides)] * b_fwd[Tensor::nd_to_flat(&nd, &b_strides)];
                }
                let out_id = self.alloc(out_shape.clone(), out_data);
                let backward_fn = Box::new(move |tensors: &mut [Tensor]| {
                    let out_grad = tensors[out_id].grad.as_cpu().clone();
                    for i in 0..out_size {
                        let nd = Tensor::flat_to_nd(i, &out_shape);
                        let idx_a = Tensor::nd_to_flat(&nd, &a_strides);
                        let idx_b = Tensor::nd_to_flat(&nd, &b_strides);
                        tensors[a_id].grad.as_cpu_mut()[idx_a] += b_fwd[idx_b] * out_grad[i];
                        tensors[b_id].grad.as_cpu_mut()[idx_b] += a_fwd[idx_a] * out_grad[i];
                    }
                });
                self.tape.nodes.push(TapeNode { inputs: vec![a_id, b_id], output: out_id, backward_fn });
                out_id
            }
        }
    }

    pub fn matmul(&mut self, a_id: usize, b_id: usize) -> usize {
        let a_shape = self.tensors[a_id].shape.clone();
        let b_shape = self.tensors[b_id].shape.clone();
        
        let m = a_shape[0];
        let k = a_shape[1];
        let n = b_shape[1];

        let device = self.device.clone();

        match &device {
            Device::Gpu(_, stream) => {
                let f_fwd = self.functions.get("matmul_f32").expect("matmul_f32 not loaded").clone();
                let f_bwd_a = self.functions.get("matmul_backward_a_f32").expect("matmul_backward_a_f32 missing").clone();
                let f_bwd_b = self.functions.get("matmul_backward_b_f32").expect("matmul_backward_b_f32 missing").clone();
                let stream_clone = stream.clone();

                let out_id = self.alloc(vec![m, n], vec![0.0; m * n]);

                let a_s = match &self.tensors[a_id].data { Storage::Gpu(s) => s, _ => unreachable!() };
                let b_s = match &self.tensors[b_id].data { Storage::Gpu(s) => s, _ => unreachable!() };
                let o_s = match &self.tensors[out_id].data { Storage::Gpu(s) => s, _ => unreachable!() };

                let m_u64 = m as u64;
                let k_u64 = k as u64;
                let n_u64 = n as u64;

                let mut builder = stream.launch_builder(&f_fwd);
                builder.arg(a_s).arg(b_s).arg(o_s).arg(&m_u64).arg(&k_u64).arg(&n_u64);

                // Use 2D Thread Blocks for matrix operations! 16x16 covers 256 threads per block.
                let cfg = LaunchConfig {
                    grid_dim: ((n as u32 + 15) / 16, (m as u32 + 15) / 16, 1),
                    block_dim: (16, 16, 1),
                    shared_mem_bytes: 0,
                };
                unsafe { builder.launch(cfg) }.expect("Failed to launch matmul_f32");

                let backward_fn = Box::new(move |tensors: &mut [Tensor]| {
                    let out_grad = match &tensors[out_id].grad { Storage::Gpu(s) => s, _ => unreachable!() };
                    let a_grad = match &tensors[a_id].grad { Storage::Gpu(s) => s, _ => unreachable!() };
                    let b_grad = match &tensors[b_id].grad { Storage::Gpu(s) => s, _ => unreachable!() };
                    let a_data = match &tensors[a_id].data { Storage::Gpu(s) => s, _ => unreachable!() };
                    let b_data = match &tensors[b_id].data { Storage::Gpu(s) => s, _ => unreachable!() };

                    // dA += dC @ B.T
                    let cfg_a = LaunchConfig {
                        grid_dim: ((k as u32 + 15) / 16, (m as u32 + 15) / 16, 1),
                        block_dim: (16, 16, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut b1 = stream_clone.launch_builder(&f_bwd_a);
                    b1.arg(out_grad).arg(b_data).arg(a_grad).arg(&m_u64).arg(&k_u64).arg(&n_u64);
                    unsafe { b1.launch(cfg_a) }.expect("Failed matmul_backward_a_f32");

                    // dB += A.T @ dC
                    let cfg_b = LaunchConfig {
                        grid_dim: ((n as u32 + 15) / 16, (k as u32 + 15) / 16, 1),
                        block_dim: (16, 16, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut b2 = stream_clone.launch_builder(&f_bwd_b);
                    b2.arg(a_data).arg(out_grad).arg(b_grad).arg(&m_u64).arg(&k_u64).arg(&n_u64);
                    unsafe { b2.launch(cfg_b) }.expect("Failed matmul_backward_b_f32");
                });

                self.tape.nodes.push(TapeNode { inputs: vec![a_id, b_id], output: out_id, backward_fn });
                out_id
            }
            Device::Cpu => {
                let mut out_data = vec![0.0; m * n];
                let a_data_fwd = self.tensors[a_id].data.as_cpu().clone();
                let b_data_fwd = self.tensors[b_id].data.as_cpu().clone();

                for r in 0..m {
                    for c in 0..n {
                        let mut sum = 0.0;
                        for i in 0..k { sum += a_data_fwd[r * k + i] * b_data_fwd[i * n + c]; }
                        out_data[r * n + c] = sum;
                    }
                }

                let out_id = self.alloc(vec![m, n], out_data);
                let backward_fn = Box::new(move |tensors: &mut [Tensor]| {
                    let out_grad = tensors[out_id].grad.as_cpu().clone();
                    {
                        let a_grad = tensors[a_id].grad.as_cpu_mut();
                        for r in 0..m {
                            for i in 0..k {
                                let mut sum = 0.0;
                                for c in 0..n { sum += out_grad[r * n + c] * b_data_fwd[i * n + c]; }
                                a_grad[r * k + i] += sum;
                            }
                        }
                    }
                    {
                        let b_grad = tensors[b_id].grad.as_cpu_mut();
                        for i in 0..k {
                            for c in 0..n {
                                let mut sum = 0.0;
                                for r in 0..m { sum += a_data_fwd[r * k + i] * out_grad[r * n + c]; }
                                b_grad[i * n + c] += sum;
                            }
                        }
                    }
                });
                self.tape.nodes.push(TapeNode { inputs: vec![a_id, b_id], output: out_id, backward_fn });
                out_id
            }
        }
    }

    pub fn silu(&mut self, a_id: usize) -> usize {
        let device = self.device.clone();
        match &device {
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
                self.tape.nodes.push(TapeNode { inputs: vec![a_id], output: out_id, backward_fn });
                out_id
            }
            Device::Gpu(_, stream) => {
                let a_shape = self.tensors[a_id].shape.clone();
                let out_size = a_shape.iter().product::<usize>();
                
                let f_fwd = self.functions.get("silu_f32").expect("silu_f32 not loaded").clone();
                let f_bwd = self.functions.get("silu_backward_f32").expect("silu_bwd not loaded").clone();
                let stream_clone = stream.clone();

                let out_id = self.alloc(a_shape, vec![0.0; out_size]);
                
                let a_s = match &self.tensors[a_id].data { Storage::Gpu(s) => s, _ => unreachable!() };
                let o_s = match &self.tensors[out_id].data { Storage::Gpu(s) => s, _ => unreachable!() };
                
                let n = out_size as u64;
                let mut builder = stream.launch_builder(&f_fwd);
                builder.arg(a_s).arg(o_s).arg(&n);
                unsafe { builder.launch(LaunchConfig::for_num_elems(out_size as u32)) }.expect("Failed to launch silu_f32");

                let backward_fn = Box::new(move |tensors: &mut [Tensor]| {
                    let a_data = match &tensors[a_id].data { Storage::Gpu(s) => s, _ => unreachable!() };
                    let out_grad = match &tensors[out_id].grad { Storage::Gpu(s) => s, _ => unreachable!() };
                    let a_grad = match &tensors[a_id].grad { Storage::Gpu(s) => s, _ => unreachable!() };

                    let mut b1 = stream_clone.launch_builder(&f_bwd);
                    b1.arg(a_data).arg(out_grad).arg(a_grad).arg(&n);
                    unsafe { b1.launch(LaunchConfig::for_num_elems(out_size as u32)) }.expect("Failed silu_backward_f32");
                });

                self.tape.nodes.push(TapeNode { inputs: vec![a_id], output: out_id, backward_fn });
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
                let out_id = self.alloc(vec![], vec![0.0]); // Loss is a scalar!

                let f_fwd = self.functions.get("cross_entropy_f32").expect("cross_entropy_f32 missing").clone();
                let f_bwd = self.functions.get("cross_entropy_backward_f32").expect("cross_entropy_backward_f32 missing").clone();
                let stream_clone = stream.clone();

                // Move targets to VRAM natively. They will be owned by the closure!
                let targets_f32: Vec<f32> = targets.iter().map(|&x| x as f32).collect();
                let targets_d = stream.clone_htod(targets_f32.as_slice()).expect("Failed HTOD targets");

                let l_s = match &self.tensors[logits_id].data { Storage::Gpu(s) => s, _ => unreachable!() };
                let o_s = match &self.tensors[out_id].data { Storage::Gpu(s) => s, _ => unreachable!() };

                let b_u64 = batch_size as u64;
                let c_u64 = num_classes as u64;

                let mut builder = stream.launch_builder(&f_fwd);
                builder.arg(l_s).arg(&targets_d).arg(o_s).arg(&b_u64).arg(&c_u64);
                unsafe { builder.launch(LaunchConfig::for_num_elems(batch_size as u32)) }.expect("Failed cross_entropy");

                let backward_fn = Box::new(move |tensors: &mut [Tensor]| {
                    let l_data = match &tensors[logits_id].data { Storage::Gpu(s) => s, _ => unreachable!() };
                    let out_grad = match &tensors[out_id].grad { Storage::Gpu(s) => s, _ => unreachable!() };
                    let l_grad = match &tensors[logits_id].grad { Storage::Gpu(s) => s, _ => unreachable!() };

                    let mut b1 = stream_clone.launch_builder(&f_bwd);
                    // The closure takes ownership of `targets_d` so it lives in VRAM as long as the tape needs it!
                    b1.arg(l_data).arg(&targets_d).arg(out_grad).arg(l_grad).arg(&b_u64).arg(&c_u64);
                    unsafe { b1.launch(LaunchConfig::for_num_elems(batch_size as u32)) }.expect("Failed cross_entropy_bwd");
                });

                self.tape.nodes.push(TapeNode { inputs: vec![logits_id], output: out_id, backward_fn });
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
                    for c in 0..num_classes { max_val = max_val.max(logits_data[b * num_classes + c]); }
                    let mut sum_exp = 0.0;
                    for c in 0..num_classes {
                        let exp = (logits_data[b * num_classes + c] - max_val).exp();
                        probs[b * num_classes + c] = exp;
                        sum_exp += exp;
                    }
                    for c in 0..num_classes { probs[b * num_classes + c] /= sum_exp; }
                    out_loss += -(probs[b * num_classes + targets[b]] + 1e-8).ln();
                }

                let out_id = self.alloc(vec![], vec![out_loss / batch_size as f32]);
                let targets_cap = targets.to_vec();
                let backward_fn = Box::new(move |tensors: &mut [Tensor]| {
                    let o_grad = tensors[out_id].grad.as_cpu()[0];
                    let l_grad = tensors[logits_id].grad.as_cpu_mut();
                    for b in 0..batch_size {
                        for c in 0..num_classes {
                            let mut g = probs[b * num_classes + c];
                            if c == targets_cap[b] { g -= 1.0; }
                            l_grad[b * num_classes + c] += (g / batch_size as f32) * o_grad;
                        }
                    }
                });
                self.tape.nodes.push(TapeNode { inputs: vec![logits_id], output: out_id, backward_fn });
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
                let num_indices = idx.data.len(); // Storage::len
                let out_size = num_indices * hidden_dim;

                let mut out_shape = idx.shape.clone();
                out_shape.push(hidden_dim);

                let f_fwd = self.functions.get("gather_f32").expect("gather_f32 not loaded").clone();
                let f_bwd = self.functions.get("gather_backward_f32").expect("gather_bwd not loaded").clone();
                let stream_clone = stream.clone();

                let out_id = self.alloc(out_shape, vec![0.0; out_size]);

                let w_s = match &self.tensors[weights_id].data { Storage::Gpu(s) => s, _ => unreachable!() };
                let idx_s = match &self.tensors[indices_id].data { Storage::Gpu(s) => s, _ => unreachable!() };
                let o_s = match &self.tensors[out_id].data { Storage::Gpu(s) => s, _ => unreachable!() };

                let hidden_u64 = hidden_dim as u64;
                let out_size_u64 = out_size as u64;

                let mut builder = stream.launch_builder(&f_fwd);
                builder.arg(w_s).arg(idx_s).arg(o_s).arg(&hidden_u64).arg(&out_size_u64);
                unsafe { builder.launch(LaunchConfig::for_num_elems(out_size as u32)) }.expect("Failed to launch gather");

                let backward_fn = Box::new(move |tensors: &mut [Tensor]| {
                    let idx_data = match &tensors[indices_id].data { Storage::Gpu(s) => s, _ => unreachable!() };
                    let out_grad = match &tensors[out_id].grad { Storage::Gpu(s) => s, _ => unreachable!() };
                    let w_grad = match &tensors[weights_id].grad { Storage::Gpu(s) => s, _ => unreachable!() };

                    let mut b1 = stream_clone.launch_builder(&f_bwd);
                    b1.arg(idx_data).arg(out_grad).arg(w_grad).arg(&hidden_u64).arg(&out_size_u64);
                    unsafe { b1.launch(LaunchConfig::for_num_elems(out_size as u32)) }.expect("Failed gather_backward");
                });

                self.tape.nodes.push(TapeNode { inputs: vec![weights_id, indices_id], output: out_id, backward_fn });
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
                    for d in 0..hidden_dim { out_data[i * hidden_dim + d] = w_data[row * hidden_dim + d]; }
                }

                let mut out_shape = idx.shape.clone();
                out_shape.push(hidden_dim);
                let out_id = self.alloc(out_shape, out_data);
                let backward_fn = Box::new(move |tensors: &mut [Tensor]| {
                    let o_grad = tensors[out_id].grad.as_cpu().clone();
                    let w_grad = tensors[weights_id].grad.as_cpu_mut();
                    for i in 0..num_indices {
                        let row = idx_data[i];
                        for d in 0..hidden_dim { w_grad[row * hidden_dim + d] += o_grad[i * hidden_dim + d]; }
                    }
                });
                self.tape.nodes.push(TapeNode { inputs: vec![weights_id, indices_id], output: out_id, backward_fn });
                out_id
            }
        }
    }

    pub fn rms_norm(&mut self, x_id: usize, weight_id: usize, eps: f32) -> usize {
        let device = self.device.clone();
        match &device {
            Device::Gpu(_, stream) => {
                let x = &self.tensors[x_id];
                let dim = *x.shape.last().unwrap();
                let num_vecs = x.data.len() / dim;
                let out_size = x.data.len();

                let f_fwd = self.functions.get("rmsnorm_f32").expect("rmsnorm_f32 missing").clone();
                let f_bwd = self.functions.get("rmsnorm_backward_f32").expect("rmsnorm_bwd missing").clone();
                let stream_clone = stream.clone();

                let out_id = self.alloc(x.shape.clone(), vec![0.0; out_size]);

                let x_s = match &self.tensors[x_id].data { Storage::Gpu(s) => s, _ => unreachable!() };
                let w_s = match &self.tensors[weight_id].data { Storage::Gpu(s) => s, _ => unreachable!() };
                let o_s = match &self.tensors[out_id].data { Storage::Gpu(s) => s, _ => unreachable!() };

                let dim_u64 = dim as u64;
                let num_vecs_u64 = num_vecs as u64;

                let mut builder = stream.launch_builder(&f_fwd);
                builder.arg(x_s).arg(w_s).arg(o_s).arg(&dim_u64).arg(&eps).arg(&num_vecs_u64);
                unsafe { builder.launch(LaunchConfig::for_num_elems(num_vecs as u32)) }.expect("Failed to launch rmsnorm");

                let backward_fn = Box::new(move |tensors: &mut [Tensor]| {
                    let x_data = match &tensors[x_id].data { Storage::Gpu(s) => s, _ => unreachable!() };
                    let w_data = match &tensors[weight_id].data { Storage::Gpu(s) => s, _ => unreachable!() };
                    let out_grad = match &tensors[out_id].grad { Storage::Gpu(s) => s, _ => unreachable!() };
                    let x_grad = match &tensors[x_id].grad { Storage::Gpu(s) => s, _ => unreachable!() };
                    let w_grad = match &tensors[weight_id].grad { Storage::Gpu(s) => s, _ => unreachable!() };

                    let mut b1 = stream_clone.launch_builder(&f_bwd);
                    b1.arg(x_data).arg(w_data).arg(out_grad).arg(x_grad).arg(w_grad)
                      .arg(&dim_u64).arg(&eps).arg(&num_vecs_u64);
                    
                    unsafe { b1.launch(LaunchConfig::for_num_elems(num_vecs as u32)) }.expect("Failed rmsnorm_bwd");
                });

                self.tape.nodes.push(TapeNode { inputs: vec![x_id, weight_id], output: out_id, backward_fn });
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
                    for d in 0..dim { ss += x_fwd[off + d].powi(2); }
                    let rrms = 1.0 / (ss / dim as f32 + eps).sqrt();
                    rrms_cache[n] = rrms;
                    for d in 0..dim { out_data[off + d] = x_fwd[off + d] * rrms * w_fwd[d]; }
                }

                let out_id = self.alloc(x.shape.clone(), out_data);
                let backward_fn = Box::new(move |tensors: &mut [Tensor]| {
                    let o_grad = tensors[out_id].grad.as_cpu().clone();
                    for n in 0..num_vecs {
                        let off = n * dim;
                        let rrms = rrms_cache[n];
                        let mut gdxw = 0.0;
                        for d in 0..dim { gdxw += o_grad[off+d] * x_fwd[off+d] * w_fwd[d]; }
                        let rrc_d = rrms.powi(3) / dim as f32;
                        for d in 0..dim {
                            let dx = rrms * (o_grad[off+d] * w_fwd[d]) - x_fwd[off+d] * rrc_d * gdxw;
                            tensors[x_id].grad.as_cpu_mut()[off+d] += dx;
                            tensors[weight_id].grad.as_cpu_mut()[d] += o_grad[off+d] * x_fwd[off+d] * rrms;
                        }
                    }
                });
                self.tape.nodes.push(TapeNode { inputs: vec![x_id, weight_id], output: out_id, backward_fn });
                out_id
            }
        }
    }

    pub fn reshape(&mut self, a_id: usize, new_shape: Vec<usize>) -> usize {
        let device = self.device.clone();
        match &device {
            Device::Gpu(_, stream) => {
                let old_size = self.tensors[a_id].data.len();
                let f_fwd = self.functions.get("copy_f32").expect("copy_f32 missing").clone();
                let f_bwd = self.functions.get("accumulate_f32").expect("accumulate_f32 missing").clone();
                let stream_clone = stream.clone();

                let out_id = self.alloc(new_shape, vec![0.0; old_size]);
                
                let a_s = match &self.tensors[a_id].data { Storage::Gpu(s) => s, _ => unreachable!() };
                let o_s = match &self.tensors[out_id].data { Storage::Gpu(s) => s, _ => unreachable!() };
                
                let n = old_size as u64;
                let mut builder = stream.launch_builder(&f_fwd);
                builder.arg(a_s).arg(o_s).arg(&n);
                unsafe { builder.launch(LaunchConfig::for_num_elems(old_size as u32)) }.expect("Failed to launch copy_f32");

                let backward_fn = Box::new(move |tensors: &mut [Tensor]| {
                    let out_grad = match &tensors[out_id].grad { Storage::Gpu(s) => s, _ => unreachable!() };
                    let a_grad = match &tensors[a_id].grad { Storage::Gpu(s) => s, _ => unreachable!() };
                    
                    let rank = 1u64; // Using 1D fallback to copy the flat gradient buffer
                    let s0 = n; let s1 = 1u64; let s2 = 1u64;
                    let a_str0 = 1u64; let a_str1 = 1u64; let a_str2 = 1u64;

                    let mut b1 = stream_clone.launch_builder(&f_bwd);
                    b1.arg(a_grad).arg(out_grad).arg(&n).arg(&rank)
                      .arg(&s0).arg(&s1).arg(&s2)
                      .arg(&a_str0).arg(&a_str1).arg(&a_str2);
                    unsafe { b1.launch(LaunchConfig::for_num_elems(old_size as u32)) }.expect("Failed accumulate reshape");
                });

                self.tape.nodes.push(TapeNode { inputs: vec![a_id], output: out_id, backward_fn });
                out_id
            }
            Device::Cpu => {
                let old_size = self.tensors[a_id].data.as_cpu().len();
                let out_id = self.alloc(new_shape, self.tensors[a_id].data.as_cpu().clone());
                let backward_fn = Box::new(move |tensors: &mut [Tensor]| {
                    let o_grad = tensors[out_id].grad.as_cpu().clone();
                    let a_grad = tensors[a_id].grad.as_cpu_mut();
                    for i in 0..old_size { a_grad[i] += o_grad[i]; }
                });
                self.tape.nodes.push(TapeNode { inputs: vec![a_id], output: out_id, backward_fn });
                out_id
            }
        }
    }

    pub fn transpose(&mut self, a_id: usize, dim0: usize, dim1: usize) -> usize {
        match &self.device {
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
                        tensors[a_id].grad.as_cpu_mut()[Tensor::nd_to_flat(&nd, &out_strides_cap)] += o_grad[i];
                    }
                });
                self.tape.nodes.push(TapeNode { inputs: vec![a_id], output: out_id, backward_fn });
                out_id
            }
            Device::Gpu(..) => panic!("GPU Transpose not yet implemented"),
        }
    }

    pub fn sum(&mut self, a_id: usize, dim: usize) -> usize {
        let device = self.device.clone();
        match &device {
            Device::Gpu(_, stream) => {
                let a = &self.tensors[a_id];
                let mut out_shape = a.shape.clone();
                let mut in_strides = a.strides.clone();
                
                let reduced_dim_size = a.shape[dim] as u64;
                let reduced_dim_stride = a.strides[dim] as u64;
                
                out_shape.remove(dim);
                in_strides.remove(dim);
                
                let out_size = if out_shape.is_empty() { 1 } else { out_shape.iter().product() };
                
                let rank = out_shape.len() as u64;
                assert!(rank <= 3, "GPU Sum supports max output rank 3");
                
                let mut os = [1u64; 3];
                let mut is = [0u64; 3];
                for i in 0..out_shape.len() {
                    os[i] = out_shape[i] as u64;
                    is[i] = in_strides[i] as u64;
                }

                let f_fwd = self.functions.get("sum_f32").expect("sum_f32 missing").clone();
                let f_bwd = self.functions.get("sum_backward_f32").expect("sum_backward_f32 missing").clone();
                let stream_clone = stream.clone();

                let out_id = self.alloc(out_shape.clone(), vec![0.0; out_size]);

                let a_s = match &self.tensors[a_id].data { Storage::Gpu(s) => s, _ => unreachable!() };
                let o_s = match &self.tensors[out_id].data { Storage::Gpu(s) => s, _ => unreachable!() };

                let out_size_u64 = out_size as u64;
                let mut builder = stream.launch_builder(&f_fwd);
                builder.arg(a_s).arg(o_s)
                       .arg(&out_size_u64).arg(&reduced_dim_size).arg(&reduced_dim_stride)
                       .arg(&rank).arg(&os[0]).arg(&os[1]).arg(&os[2])
                       .arg(&is[0]).arg(&is[1]).arg(&is[2]);
                unsafe { builder.launch(LaunchConfig::for_num_elems(out_size as u32)) }.expect("Failed to launch sum_f32");

                let backward_fn = Box::new(move |tensors: &mut [Tensor]| {
                    let out_grad = match &tensors[out_id].grad { Storage::Gpu(s) => s, _ => unreachable!() };
                    let a_grad = match &tensors[a_id].grad { Storage::Gpu(s) => s, _ => unreachable!() };

                    let mut b1 = stream_clone.launch_builder(&f_bwd);
                    b1.arg(out_grad).arg(a_grad)
                      .arg(&out_size_u64).arg(&reduced_dim_size).arg(&reduced_dim_stride)
                      .arg(&rank).arg(&os[0]).arg(&os[1]).arg(&os[2])
                      .arg(&is[0]).arg(&is[1]).arg(&is[2]);
                    unsafe { b1.launch(LaunchConfig::for_num_elems(out_size as u32)) }.expect("Failed sum_bwd");
                });

                self.tape.nodes.push(TapeNode { inputs: vec![a_id], output: out_id, backward_fn });
                out_id
            }
            Device::Cpu => {
                let a = &self.tensors[a_id];
                let mut out_shape = a.shape.clone();
                out_shape.remove(dim);
                let out_size = if out_shape.is_empty() { 1 } else { out_shape.iter().product() };
                let mut out_data = vec![0.0; out_size];
                let os = Tensor::compute_strides(&out_shape);
                let a_shape = a.shape.clone();
                let a_data = a.data.as_cpu();
                for i in 0..a_data.len() {
                    let mut nd = Tensor::flat_to_nd(i, &a_shape);
                    nd.remove(dim);
                    let idx = if out_shape.is_empty() { 0 } else { Tensor::nd_to_flat(&nd, &os) };
                    out_data[idx] += a_data[i];
                }
                let out_id = self.alloc(out_shape, out_data);
                let backward_fn = Box::new(move |tensors: &mut [Tensor]| {
                    let o_grad = tensors[out_id].grad.as_cpu().clone();
                    for i in 0..tensors[a_id].data.as_cpu().len() {
                        let mut nd = Tensor::flat_to_nd(i, &a_shape);
                        nd.remove(dim);
                        let idx = if tensors[out_id].shape.is_empty() { 0 } else { Tensor::nd_to_flat(&nd, &os) };
                        tensors[a_id].grad.as_cpu_mut()[i] += o_grad[idx];
                    }
                });
                self.tape.nodes.push(TapeNode { inputs: vec![a_id], output: out_id, backward_fn });
                out_id
            }
        }
    }

    pub fn max(&mut self, a_id: usize, dim: usize) -> usize {
        let device = self.device.clone();
        match &device {
            Device::Gpu(_, stream) => {
                let a = &self.tensors[a_id];
                let mut out_shape = a.shape.clone();
                let mut in_strides = a.strides.clone();
                
                let reduced_dim_size = a.shape[dim] as u64;
                let reduced_dim_stride = a.strides[dim] as u64;
                
                out_shape.remove(dim);
                in_strides.remove(dim);
                
                let out_size = if out_shape.is_empty() { 1 } else { out_shape.iter().product() };
                let rank = out_shape.len() as u64;
                
                let mut os = [1u64; 3];
                let mut is = [0u64; 3];
                for i in 0..out_shape.len() {
                    os[i] = out_shape[i] as u64;
                    is[i] = in_strides[i] as u64;
                }

                let f_fwd = self.functions.get("max_f32").expect("max_f32 missing").clone();
                let f_bwd = self.functions.get("max_backward_f32").expect("max_backward_f32 missing").clone();
                let stream_clone = stream.clone();

                let out_id = self.alloc(out_shape.clone(), vec![0.0; out_size]);

                let a_s = match &self.tensors[a_id].data { Storage::Gpu(s) => s, _ => unreachable!() };
                let o_s = match &self.tensors[out_id].data { Storage::Gpu(s) => s, _ => unreachable!() };

                let out_size_u64 = out_size as u64;
                let mut builder = stream.launch_builder(&f_fwd);
                builder.arg(a_s).arg(o_s)
                       .arg(&out_size_u64).arg(&reduced_dim_size).arg(&reduced_dim_stride)
                       .arg(&rank).arg(&os[0]).arg(&os[1]).arg(&os[2])
                       .arg(&is[0]).arg(&is[1]).arg(&is[2]);
                unsafe { builder.launch(LaunchConfig::for_num_elems(out_size as u32)) }.expect("Failed max_f32");

                let backward_fn = Box::new(move |tensors: &mut [Tensor]| {
                    let a_data = match &tensors[a_id].data { Storage::Gpu(s) => s, _ => unreachable!() };
                    let out_grad = match &tensors[out_id].grad { Storage::Gpu(s) => s, _ => unreachable!() };
                    let a_grad = match &tensors[a_id].grad { Storage::Gpu(s) => s, _ => unreachable!() };

                    let mut b1 = stream_clone.launch_builder(&f_bwd);
                    b1.arg(a_data).arg(out_grad).arg(a_grad)
                      .arg(&out_size_u64).arg(&reduced_dim_size).arg(&reduced_dim_stride)
                      .arg(&rank).arg(&os[0]).arg(&os[1]).arg(&os[2])
                      .arg(&is[0]).arg(&is[1]).arg(&is[2]);
                    unsafe { b1.launch(LaunchConfig::for_num_elems(out_size as u32)) }.expect("Failed max_bwd");
                });

                self.tape.nodes.push(TapeNode { inputs: vec![a_id], output: out_id, backward_fn });
                out_id
            }
            Device::Cpu => {
                let a = &self.tensors[a_id];
                let mut out_shape = a.shape.clone();
                out_shape.remove(dim);
                let out_size = if out_shape.is_empty() { 1 } else { out_shape.iter().product() };
                let mut out_data = vec![f32::NEG_INFINITY; out_size];
                let mut argmax = vec![0; out_size];
                let os = Tensor::compute_strides(&out_shape);
                let a_data = a.data.as_cpu();
                for i in 0..a_data.len() {
                    let mut nd = Tensor::flat_to_nd(i, &a.shape);
                    nd.remove(dim);
                    let idx = if out_shape.is_empty() { 0 } else { Tensor::nd_to_flat(&nd, &os) };
                    if a_data[i] > out_data[idx] {
                        out_data[idx] = a_data[i];
                        argmax[idx] = i;
                    }
                }
                let out_id = self.alloc(out_shape, out_data);
                let backward_fn = Box::new(move |tensors: &mut [Tensor]| {
                    let o_grad = tensors[out_id].grad.as_cpu().clone();
                    for i in 0..o_grad.len() { tensors[a_id].grad.as_cpu_mut()[argmax[i]] += o_grad[i]; }
                });
                self.tape.nodes.push(TapeNode { inputs: vec![a_id], output: out_id, backward_fn });
                out_id
            }
        }
    }

    pub fn backward(&mut self, loss_id: usize) {
        match &self.device {
            Device::Cpu => {
                let loss_grad = self.tensors[loss_id].grad.as_cpu_mut();
                for g in loss_grad.iter_mut() { *g = 1.0; }
            }
            Device::Gpu(_, stream) => {
                let grad_slice = match &self.tensors[loss_id].grad {
                    Storage::Gpu(s) => s,
                    _ => unreachable!(),
                };
                let f = self.functions.get("fill_f32").expect("fill_f32 not loaded").clone();
                let n = grad_slice.len() as u64;
                let val = 1.0f32;
                let mut builder = stream.launch_builder(&f);
                builder.arg(grad_slice).arg(&val).arg(&n);
                unsafe { builder.launch(LaunchConfig::for_num_elems(n as u32)) }.expect("Failed to fill seed gradient");
            }
        }
        for node in self.tape.nodes.iter().rev() { (node.backward_fn)(&mut self.tensors); }
    }
}
