use crate::tape::{Tape, TapeNode};
use crate::tensor::{Tensor, Device};

#[derive(Default)]
pub struct Graph {
    pub tensors: Vec<Tensor>,
    pub tape: Tape,
    pub device: Device,
}

impl Graph {
    pub fn new(device: Device) -> Self {
        Self {
            tensors: Vec::new(),
            tape: Tape::default(),
            device,
        }
    }

    pub fn alloc(&mut self, shape: Vec<usize>, data: Vec<f32>) -> usize {
        let id = self.tensors.len();
        let tensor = Tensor::new(id, shape, data, self.device.clone());
        self.tensors.push(tensor);
        id
    }

    /// Element-wise addition with broadcasting
    pub fn add(&mut self, a_id: usize, b_id: usize) -> usize {
        let out_shape = Tensor::get_broadcasted_shape(&self.tensors[a_id].shape, &self.tensors[b_id].shape);
        let out_size: usize = out_shape.iter().product();
        
        let a_strides = Tensor::get_broadcasted_strides(&self.tensors[a_id].shape, &self.tensors[a_id].strides, &out_shape);
        let b_strides = Tensor::get_broadcasted_strides(&self.tensors[b_id].shape, &self.tensors[b_id].strides, &out_shape);

        let mut out_data = vec![0.0; out_size];

        {
            let a = &self.tensors[a_id];
            let b = &self.tensors[b_id];
            let a_data = a.data.as_cpu(); // Using our new helper
            let b_data = b.data.as_cpu();

            for i in 0..out_size {
                let nd = Tensor::flat_to_nd(i, &out_shape);
                let idx_a = Tensor::nd_to_flat(&nd, &a_strides);
                let idx_b = Tensor::nd_to_flat(&nd, &b_strides);
                out_data[i] = a_data[idx_a] + b_data[idx_b];
            }
        }

        let out_id = self.alloc(out_shape.clone(), out_data);

        let backward_fn = Box::new(move |tensors: &mut [Tensor]| {
            let out_grad = tensors[out_id].grad.as_cpu().clone();
            for i in 0..out_size {
                let nd = Tensor::flat_to_nd(i, &out_shape);
                let idx_a = Tensor::nd_to_flat(&nd, &a_strides);
                let idx_b = Tensor::nd_to_flat(&nd, &b_strides);
                tensors[a_id].grad.as_cpu_mut()[idx_a] += out_grad[i];
                tensors[b_id].grad.as_cpu_mut()[idx_b] += out_grad[i];
            }
        });

        self.tape.nodes.push(TapeNode {
            inputs: vec![a_id, b_id],
            output: out_id,
            backward_fn,
        });

        out_id
    }

    /// Element-wise multiplication with broadcasting
    pub fn mul(&mut self, a_id: usize, b_id: usize) -> usize {
        let a = &self.tensors[a_id];
        let b = &self.tensors[b_id];

        let out_shape = Tensor::get_broadcasted_shape(&a.shape, &b.shape);
        let a_strides = Tensor::get_broadcasted_strides(&a.shape, &a.strides, &out_shape);
        let b_strides = Tensor::get_broadcasted_strides(&b.shape, &b.strides, &out_shape);

        let out_size: usize = out_shape.iter().product();
        let mut out_data = vec![0.0; out_size];

        for i in 0..out_size {
            let nd = Tensor::flat_to_nd(i, &out_shape);
            let idx_a = Tensor::nd_to_flat(&nd, &a_strides);
            let idx_b = Tensor::nd_to_flat(&nd, &b_strides);
            out_data[i] = a.data[idx_a] * b.data[idx_b];
        }

        let a_data = a.data.clone();
        let b_data = b.data.clone();

        let out_id = self.alloc(out_shape.clone(), out_data);

        let backward_fn = Box::new(move |tensors: &mut [Tensor]| {
            let out_grad = tensors[out_id].grad.clone();
            for i in 0..out_size {
                let nd = Tensor::flat_to_nd(i, &out_shape);
                let idx_a = Tensor::nd_to_flat(&nd, &a_strides);
                let idx_b = Tensor::nd_to_flat(&nd, &b_strides);
                tensors[a_id].grad[idx_a] += b_data[idx_b] * out_grad[i];
                tensors[b_id].grad[idx_b] += a_data[idx_a] * out_grad[i];
            }
        });

        self.tape.nodes.push(TapeNode {
            inputs: vec![a_id, b_id],
            output: out_id,
            backward_fn,
        });

        out_id
    }

    /// Transposes a tensor by swapping two dimensions (Eager copy for CPU, stride manipulation natively)
    pub fn transpose(&mut self, a_id: usize, dim0: usize, dim1: usize) -> usize {
        let a = &self.tensors[a_id];
        let mut out_shape = a.shape.clone();
        out_shape.swap(dim0, dim1);

        let mut out_strides = a.strides.clone();
        out_strides.swap(dim0, dim1); // Swap original strides to map ND index back to physical flat memory

        let out_size = a.data.len();
        let mut out_data = vec![0.0; out_size];

        for i in 0..out_size {
            let nd = Tensor::flat_to_nd(i, &out_shape);
            // By using swapped strides, we read the exact transposed value from the flat array!
            let a_flat = Tensor::nd_to_flat(&nd, &out_strides); 
            out_data[i] = a.data[a_flat];
        }

        let out_id = self.alloc(out_shape.clone(), out_data);
        let out_strides_cap = out_strides.clone();

        let backward_fn = Box::new(move |tensors: &mut [Tensor]| {
            let out_grad = tensors[out_id].grad.clone();
            for i in 0..out_size {
                let nd = Tensor::flat_to_nd(i, &out_shape);
                let a_flat = Tensor::nd_to_flat(&nd, &out_strides_cap);
                tensors[a_id].grad[a_flat] += out_grad[i];
            }
        });

        self.tape.nodes.push(TapeNode { inputs: vec![a_id], output: out_id, backward_fn });
        out_id
    }

    /// Sums elements along a specific dimension (keepdim = false)
    pub fn sum(&mut self, a_id: usize, dim: usize) -> usize {
        let a = &self.tensors[a_id];
        let mut out_shape = a.shape.clone();
        out_shape.remove(dim);
        
        let out_size: usize = if out_shape.is_empty() { 1 } else { out_shape.iter().product() };
        let mut out_data = vec![0.0; out_size];
        let out_strides = Tensor::compute_strides(&out_shape);

        for i in 0..a.data.len() {
            let mut nd = Tensor::flat_to_nd(i, &a.shape);
            nd.remove(dim);
            let out_flat = if out_shape.is_empty() { 0 } else { Tensor::nd_to_flat(&nd, &out_strides) };
            out_data[out_flat] += a.data[i];
        }

        let a_shape = a.shape.clone();
        let out_id = self.alloc(out_shape.clone(), out_data);

        let backward_fn = Box::new(move |tensors: &mut [Tensor]| {
            let out_grad = tensors[out_id].grad.clone();
            for i in 0..tensors[a_id].data.len() {
                let mut nd = Tensor::flat_to_nd(i, &a_shape);
                nd.remove(dim);
                let out_flat = if out_shape.is_empty() { 0 } else { Tensor::nd_to_flat(&nd, &out_strides) };
                tensors[a_id].grad[i] += out_grad[out_flat]; // Broadcast gradient back
            }
        });

        self.tape.nodes.push(TapeNode { inputs: vec![a_id], output: out_id, backward_fn });
        out_id
    }

    /// Max operation along a specific dimension (keepdim = false)
    pub fn max(&mut self, a_id: usize, dim: usize) -> usize {
        let a = &self.tensors[a_id];
        let mut out_shape = a.shape.clone();
        out_shape.remove(dim);
        
        let out_size: usize = if out_shape.is_empty() { 1 } else { out_shape.iter().product() };
        let mut out_data = vec![std::f32::NEG_INFINITY; out_size];
        let mut argmax = vec![0; out_size]; // store flat index of 'a' for backward routing
        
        let out_strides = Tensor::compute_strides(&out_shape);

        for i in 0..a.data.len() {
            let mut nd = Tensor::flat_to_nd(i, &a.shape);
            nd.remove(dim);
            let out_flat = if out_shape.is_empty() { 0 } else { Tensor::nd_to_flat(&nd, &out_strides) };
            
            let val = a.data[i];
            if val > out_data[out_flat] {
                out_data[out_flat] = val;
                argmax[out_flat] = i;
            }
        }

        let out_id = self.alloc(out_shape, out_data);

        // BACKWARD: Only the element that was the maximum receives the gradient!
        let backward_fn = Box::new(move |tensors: &mut [Tensor]| {
            let out_grad = tensors[out_id].grad.clone();
            for i in 0..out_grad.len() {
                let max_input_idx = argmax[i];
                tensors[a_id].grad[max_input_idx] += out_grad[i];
            }
        });

        self.tape.nodes.push(TapeNode { inputs: vec![a_id], output: out_id, backward_fn });
        out_id
    }

    /// 2D Matrix Multiplication: C = A @ B
    pub fn matmul(&mut self, a_id: usize, b_id: usize) -> usize {
        let a = &self.tensors[a_id];
        let b = &self.tensors[b_id];
        
        assert_eq!(a.shape.len(), 2, "MatMul requires 2D tensors");
        assert_eq!(b.shape.len(), 2, "MatMul requires 2D tensors");
        
        let m = a.shape[0];
        let k = a.shape[1];
        let k2 = b.shape[0];
        let n = b.shape[1];
        assert_eq!(k, k2, "Inner dimensions must match for MatMul ({} != {})", k, k2);

        let mut out_data = vec![0.0; m * n];
        
        // Naive CPU MatMul
        for r in 0..m {
            for c in 0..n {
                let mut sum = 0.0;
                for i in 0..k {
                    sum += a.data[r * k + i] * b.data[i * n + c];
                }
                out_data[r * n + c] = sum;
            }
        }

        let a_data = a.data.clone();
        let b_data = b.data.clone();

        let out_id = self.alloc(vec![m, n], out_data);

        // BACKWARD: dA += dC @ B.T  and  dB += A.T @ dC
        let backward_fn = Box::new(move |tensors: &mut [Tensor]| {
            let out_grad = tensors[out_id].grad.clone();

            // grad_A += out_grad @ B.T
            for r in 0..m {
                for i in 0..k {
                    let mut sum = 0.0;
                    for c in 0..n {
                        sum += out_grad[r * n + c] * b_data[i * n + c]; // B transposed access
                    }
                    tensors[a_id].grad[r * k + i] += sum;
                }
            }

            // grad_B += A.T @ out_grad
            for i in 0..k {
                for c in 0..n {
                    let mut sum = 0.0;
                    for r in 0..m {
                        sum += a_data[r * k + i] * out_grad[r * n + c]; // A transposed access
                    }
                    tensors[b_id].grad[i * n + c] += sum;
                }
            }
        });

        self.tape.nodes.push(TapeNode {
            inputs: vec![a_id, b_id],
            output: out_id,
            backward_fn,
        });

        out_id
    }

    /// SiLU Activation: f(x) = x * sigmoid(x)
    pub fn silu(&mut self, a_id: usize) -> usize {
        let a = &self.tensors[a_id];
        let out_size = a.data.len();
        let mut out_data = vec![0.0; out_size];
        
        for i in 0..out_size {
            let x = a.data[i];
            let sig = 1.0 / (1.0 + (-x).exp());
            out_data[i] = x * sig;
        }

        let a_data = a.data.clone();
        let out_id = self.alloc(a.shape.clone(), out_data);

        // BACKWARD: f'(x) = f(x) + sigmoid(x) * (1 - f(x))
        let backward_fn = Box::new(move |tensors: &mut [Tensor]| {
            let out_grad = tensors[out_id].grad.clone();
            for i in 0..out_size {
                let x = a_data[i];
                let sig = 1.0 / (1.0 + (-x).exp());
                let silu_val = x * sig;
                let grad_in = silu_val + sig * (1.0 - silu_val);
                tensors[a_id].grad[i] += out_grad[i] * grad_in;
            }
        });

        self.tape.nodes.push(TapeNode { inputs: vec![a_id], output: out_id, backward_fn });
        out_id
    }

    /// Fused Softmax + Cross Entropy Loss
    /// logits: [batch_size, num_classes]
    /// targets: [batch_size] containing the ground-truth class indices
    pub fn cross_entropy(&mut self, logits_id: usize, targets: &[usize]) -> usize {
        let logits = &self.tensors[logits_id];
        assert_eq!(logits.shape.len(), 2, "Logits must be 2D [batch, classes]");
        let batch_size = logits.shape[0];
        let num_classes = logits.shape[1];
        assert_eq!(targets.len(), batch_size, "Targets length must match batch size");

        let mut out_loss = 0.0;
        let mut probs = vec![0.0; batch_size * num_classes];

        for b in 0..batch_size {
            // 1. Find max for numerical stability (prevent exp() overflow)
            let mut max_val = std::f32::NEG_INFINITY;
            for c in 0..num_classes {
                let val = logits.data[b * num_classes + c];
                if val > max_val { max_val = val; }
            }

            // 2. Compute exponentials and sum
            let mut sum_exp = 0.0;
            for c in 0..num_classes {
                let exp_val = (logits.data[b * num_classes + c] - max_val).exp();
                probs[b * num_classes + c] = exp_val;
                sum_exp += exp_val;
            }

            // 3. Normalize to get probabilities and compute negative log-likelihood
            let target_idx = targets[b];
            for c in 0..num_classes {
                probs[b * num_classes + c] /= sum_exp;
            }
            
            // Loss = -log(prob of the true class)
            let true_prob = probs[b * num_classes + target_idx];
            // Add tiny epsilon to prevent log(0)
            out_loss += -(true_prob + 1e-8).ln(); 
        }

        out_loss /= batch_size as f32; // Average loss over the batch

        let out_id = self.alloc(vec![], vec![out_loss]); // Returns a scalar loss
        let targets_cap = targets.to_vec();

        // BACKWARD: An incredibly elegant gradient! 
        // dL/dLogits = (probs - 1_if_true_class) / batch_size
        let backward_fn = Box::new(move |tensors: &mut [Tensor]| {
            let out_grad = tensors[out_id].grad[0]; // Scalar gradient from the loss
            for b in 0..batch_size {
                let target_idx = targets_cap[b];
                for c in 0..num_classes {
                    let idx = b * num_classes + c;
                    let mut grad = probs[idx];
                    if c == target_idx {
                        grad -= 1.0;
                    }
                    tensors[logits_id].grad[idx] += (grad / batch_size as f32) * out_grad;
                }
            }
        });

        self.tape.nodes.push(TapeNode { inputs: vec![logits_id], output: out_id, backward_fn });
        out_id
    }

    /// Gathers rows from a weight matrix using indices.
    /// weights: [vocab_size, hidden_dim]
    /// indices: [..., num_indices] containing float representations of integers
    pub fn gather(&mut self, weights_id: usize, indices_id: usize) -> usize {
        let weights = &self.tensors[weights_id];
        let indices = &self.tensors[indices_id];
        
        assert_eq!(weights.shape.len(), 2, "Embedding weights must be 2D [vocab_size, hidden_dim]");
        let vocab_size = weights.shape[0];
        let hidden_dim = weights.shape[1];
        
        let mut out_shape = indices.shape.clone();
        out_shape.push(hidden_dim);
        
        let num_indices = indices.data.len();
        let out_size = num_indices * hidden_dim;
        let mut out_data = vec![0.0; out_size];
        
        let mut indices_usize = vec![0; num_indices];
        for i in 0..num_indices {
            let idx = indices.data[i] as usize;
            assert!(idx < vocab_size, "Index {} out of bounds for vocab size {}", idx, vocab_size);
            indices_usize[i] = idx;
            
            // Copy the row
            for d in 0..hidden_dim {
                out_data[i * hidden_dim + d] = weights.data[idx * hidden_dim + d];
            }
        }
        
        let out_id = self.alloc(out_shape, out_data);
        
        // BACKWARD: Accumulate incoming gradients into the accessed rows of the weights
        let backward_fn = Box::new(move |tensors: &mut [Tensor]| {
            let out_grad = tensors[out_id].grad.clone();
            for i in 0..num_indices {
                let idx = indices_usize[i];
                for d in 0..hidden_dim {
                    tensors[weights_id].grad[idx * hidden_dim + d] += out_grad[i * hidden_dim + d];
                }
            }
        });
        
        self.tape.nodes.push(TapeNode {
            inputs: vec![weights_id, indices_id],
            output: out_id,
            backward_fn,
        });
        
        out_id
    }

    /// Applies Root Mean Square Normalization (RMSNorm)
    /// x: [..., hidden_dim]
    /// weight: [hidden_dim]
    pub fn rms_norm(&mut self, x_id: usize, weight_id: usize, eps: f32) -> usize {
        let x = &self.tensors[x_id];
        let weight = &self.tensors[weight_id];
        
        let hidden_dim = *x.shape.last().expect("RMSNorm requires at least 1D tensor");
        assert_eq!(weight.shape, vec![hidden_dim], "RMSNorm weight must match the last dimension of x");
        
        let num_vectors = x.data.len() / hidden_dim;
        let mut out_data = vec![0.0; x.data.len()];
        let mut rrms_cache = vec![0.0; num_vectors]; // Cache the reciprocal root for the backward pass
        
        for n in 0..num_vectors {
            let offset = n * hidden_dim;
            
            // 1. Calculate Mean Squared
            let mut sum_sq = 0.0;
            for d in 0..hidden_dim {
                let val = x.data[offset + d];
                sum_sq += val * val;
            }
            let mean_sq = sum_sq / hidden_dim as f32;
            
            // 2. Calculate Reciprocal Root Mean Square
            let rrms = 1.0 / (mean_sq + eps).sqrt();
            rrms_cache[n] = rrms;
            
            // 3. Apply normalization and learnable gain
            for d in 0..hidden_dim {
                out_data[offset + d] = x.data[offset + d] * rrms * weight.data[d];
            }
        }
        
        let x_data = x.data.clone();
        let w_data = weight.data.clone();
        
        let out_id = self.alloc(x.shape.clone(), out_data);
        
        // BACKWARD: Fused Analytical Gradient for RMSNorm
        let backward_fn = Box::new(move |tensors: &mut [Tensor]| {
            let out_grad = tensors[out_id].grad.clone();
            
            for n in 0..num_vectors {
                let offset = n * hidden_dim;
                let rrms = rrms_cache[n];
                
                // Compute dot product for the gradient of the variance: sum(dL/dy * x * w)
                let mut grad_dot_x_w = 0.0;
                for d in 0..hidden_dim {
                    grad_dot_x_w += out_grad[offset + d] * x_data[offset + d] * w_data[d];
                }
                
                let rrms_cubed_over_d = (rrms * rrms * rrms) / hidden_dim as f32;
                
                for d in 0..hidden_dim {
                    let dy = out_grad[offset + d];
                    let x_val = x_data[offset + d];
                    let w_val = w_data[d];
                    
                    // Gradient w.r.t input x
                    let dx = rrms * (dy * w_val) - x_val * rrms_cubed_over_d * grad_dot_x_w;
                    tensors[x_id].grad[offset + d] += dx;
                    
                    // Gradient w.r.t learnable weight vector
                    let dw = dy * x_val * rrms;
                    tensors[weight_id].grad[d] += dw;
                }
            }
        });
        
        self.tape.nodes.push(TapeNode {
            inputs: vec![x_id, weight_id],
            output: out_id,
            backward_fn,
        });
        
        out_id
    }

    /// Reshapes a tensor. Must preserve the total number of elements.
    pub fn reshape(&mut self, a_id: usize, new_shape: Vec<usize>) -> usize {
        let a = &self.tensors[a_id];
        let old_size: usize = a.shape.iter().product();
        let new_size: usize = new_shape.iter().product();
        assert_eq!(old_size, new_size, "Reshape: total elements must match ({} != {})", old_size, new_size);

        // For CPU, we just clone the data into a new tensor with the new shape
        let out_id = self.alloc(new_shape, a.data.clone());

        // BACKWARD: The gradient of a reshape is just the output gradient reshaped back
        let backward_fn = Box::new(move |tensors: &mut [Tensor]| {
            let out_grad = tensors[out_id].grad.clone();
            for i in 0..old_size {
                tensors[a_id].grad[i] += out_grad[i];
            }
        });

        self.tape.nodes.push(TapeNode {
            inputs: vec![a_id],
            output: out_id,
            backward_fn,
        });

        out_id
    }

    /// Triggers reverse-mode automatic differentiation.
    pub fn backward(&mut self, loss_id: usize) {
        // 1. Seed the gradient of the loss output to 1.0 (dL/dL = 1)
        for i in 0..self.tensors[loss_id].grad.len() {
            self.tensors[loss_id].grad[i] = 1.0;
        }

        // 2. Walk backwards through the tape and apply the chain rule
        for node in self.tape.nodes.iter().rev() {
            (node.backward_fn)(&mut self.tensors);
        }
    }
}
