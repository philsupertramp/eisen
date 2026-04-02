// --- BROADCAST-AWARE ADDITION ---
extern "C" __global__ void add_f32(
    const float* a,
    const float* b,
    float* out,
    const size_t n,
    const size_t rank,
    const size_t s0, const size_t s1, const size_t s2,
    const size_t a0, const size_t a1, const size_t a2,
    const size_t b0, const size_t b1, const size_t b2
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        size_t temp = i;
        size_t idx_a = 0;
        size_t idx_b = 0;
        
        // Dynamically unpack flat index 'i' into n-dimensional strides
        if (rank > 2) { size_t c = temp % s2; temp /= s2; idx_a += c * a2; idx_b += c * b2; }
        if (rank > 1) { size_t c = temp % s1; temp /= s1; idx_a += c * a1; idx_b += c * b1; }
        if (rank > 0) { size_t c = temp % s0; temp /= s0; idx_a += c * a0; idx_b += c * b0; }
        
        out[i] = a[idx_a] + b[idx_b];
    }
}

// --- BROADCAST-AWARE ACCUMULATION (BACKPROP) ---
extern "C" __global__ void accumulate_f32(
    float* grad_target,
    const float* grad_out,
    const size_t n,
    const size_t rank,
    const size_t s0, const size_t s1, const size_t s2,
    const size_t t0, const size_t t1, const size_t t2
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        size_t temp = i;
        size_t idx_t = 0;
        
        if (rank > 2) { size_t c = temp % s2; temp /= s2; idx_t += c * t2; }
        if (rank > 1) { size_t c = temp % s1; temp /= s1; idx_t += c * t1; }
        if (rank > 0) { size_t c = temp % s0; temp /= s0; idx_t += c * t0; }
        
        // CRITICAL: When accumulating gradients for broadcasted tensors (like biases),
        // multiple threads will write to the same `idx_t`. We MUST use atomicAdd!
        atomicAdd(&grad_target[idx_t], grad_out[i]);
    }
}

extern "C" __global__ void fill_f32(
    float* data,
    const float value,
    const size_t n
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        data[i] = value;
    }
}

// ... mul and matmul kernels remain exactly as they were in the previous step ...
extern "C" __global__ void mul_f32(
    const float* a, const float* b, float* out, const size_t n
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { out[i] = a[i] * b[i]; }
}

extern "C" __global__ void mul_backward_f32(
    const float* a, const float* b, const float* grad_out,
    float* grad_a, float* grad_b, const size_t n
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        grad_a[i] += b[i] * grad_out[i];
        grad_b[i] += a[i] * grad_out[i];
    }
}

extern "C" __global__ void matmul_f32(
    const float* a, const float* b, float* out,
    const size_t m, const size_t k, const size_t n
) {
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m && col < n) {
        float sum = 0.0f;
        for (size_t i = 0; i < k; ++i) { sum += a[row * k + i] * b[i * n + col]; }
        out[row * n + col] = sum;
    }
}

extern "C" __global__ void matmul_backward_a_f32(
    const float* grad_out, const float* b, float* grad_a,
    const size_t m, const size_t k, const size_t n
) {
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m && col < k) {
        float sum = 0.0f;
        for (size_t i = 0; i < n; ++i) { sum += grad_out[row * n + i] * b[col * n + i]; }
        grad_a[row * k + col] += sum;
    }
}

extern "C" __global__ void matmul_backward_b_f32(
    const float* a, const float* grad_out, float* grad_b,
    const size_t m, const size_t k, const size_t n
) {
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < k && col < n) {
        float sum = 0.0f;
        for (size_t i = 0; i < m; ++i) { sum += a[i * k + row] * grad_out[i * n + col]; }
        grad_b[row * n + col] += sum;
    }
}

extern "C" __global__ void silu_f32(
    const float* x, float* out, const size_t n
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float val = x[i];
        out[i] = val / (1.0f + expf(-val));
    }
}

extern "C" __global__ void silu_backward_f32(
    const float* x, const float* grad_out, float* grad_x, const size_t n
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float val = x[i];
        float sig = 1.0f / (1.0f + expf(-val));
        float silu = val * sig;
        
        // f'(x) = f(x) + sig(x) * (1 - f(x))
        float d_silu = silu + sig * (1.0f - silu);
        grad_x[i] += grad_out[i] * d_silu;
    }
}

extern "C" __global__ void gather_f32(
    const float* weights,  // [vocab_size, hidden_dim]
    const float* indices,  // [num_indices]
    float* out,            // [num_indices, hidden_dim]
    const size_t hidden_dim,
    const size_t out_size
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < out_size) {
        size_t row = i / hidden_dim;
        size_t col = i % hidden_dim;
        
        // Cast float index back to size_t
        size_t w_row = (size_t)indices[row]; 
        
        out[i] = weights[w_row * hidden_dim + col];
    }
}

extern "C" __global__ void gather_backward_f32(
    const float* indices,  // [num_indices]
    const float* grad_out, // [num_indices, hidden_dim]
    float* grad_weights,   // [vocab_size, hidden_dim]
    const size_t hidden_dim,
    const size_t out_size
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < out_size) {
        size_t row = i / hidden_dim;
        size_t col = i % hidden_dim;
        
        size_t w_row = (size_t)indices[row];
        
        // CRITICAL: Multiple tokens in the sequence might map to the same embedding row.
        // We MUST use atomicAdd to accumulate gradients safely across threads.
        atomicAdd(&grad_weights[w_row * hidden_dim + col], grad_out[i]);
    }
}

extern "C" __global__ void rmsnorm_f32(
    const float* x, const float* w, float* out,
    const size_t dim, const float eps, const size_t num_vecs
) {
    // 1 thread per vector/row
    size_t n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n < num_vecs) {
        size_t offset = n * dim;
        float sum_sq = 0.0f;
        
        for (size_t d = 0; d < dim; ++d) {
            float val = x[offset + d];
            sum_sq += val * val;
        }
        
        // Fast reciprocal square root: 1.0 / sqrt(x)
        float rrms = rsqrtf(sum_sq / (float)dim + eps);
        
        for (size_t d = 0; d < dim; ++d) {
            out[offset + d] = x[offset + d] * rrms * w[d];
        }
    }
}

extern "C" __global__ void rmsnorm_backward_f32(
    const float* x, const float* w, const float* grad_out,
    float* grad_x, float* grad_w,
    const size_t dim, const float eps, const size_t num_vecs
) {
    // 1 thread per vector/row
    size_t n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n < num_vecs) {
        size_t offset = n * dim;
        
        // 1. Recompute the variance to save VRAM caching
        float sum_sq = 0.0f;
        for (size_t d = 0; d < dim; ++d) {
            float val = x[offset + d];
            sum_sq += val * val;
        }
        float rrms = rsqrtf(sum_sq / (float)dim + eps);
        
        // 2. Compute grad_dot_x_w
        float grad_dot_x_w = 0.0f;
        for (size_t d = 0; d < dim; ++d) {
            grad_dot_x_w += grad_out[offset + d] * x[offset + d] * w[d];
        }
        
        float rrc_d = (rrms * rrms * rrms) / (float)dim;
        
        // 3. Compute gradients
        for (size_t d = 0; d < dim; ++d) {
            float val_x = x[offset + d];
            float val_w = w[d];
            float go = grad_out[offset + d];
            
            // dx
            float dx = rrms * (go * val_w) - val_x * rrc_d * grad_dot_x_w;
            grad_x[offset + d] += dx;
            
            // dw (Many rows update the same weights, so atomicAdd is required)
            float dw = go * val_x * rrms;
            atomicAdd(&grad_w[d], dw);
        }
    }
}

extern "C" __global__ void copy_f32(
    const float* src, float* dst, const size_t n
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { dst[i] = src[i]; }
}

// --- CROSS ENTROPY LOSS ---
extern "C" __global__ void cross_entropy_f32(
    const float* logits,
    const float* targets,
    float* out_loss,
    const size_t batch_size,
    const size_t num_classes
) {
    size_t b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b < batch_size) {
        // Log-Sum-Exp Trick for numerical stability
        float max_val = -1e20f;
        for (size_t c = 0; c < num_classes; ++c) {
            float val = logits[b * num_classes + c];
            if (val > max_val) max_val = val;
        }
        float sum_exp = 0.0f;
        for (size_t c = 0; c < num_classes; ++c) {
            sum_exp += expf(logits[b * num_classes + c] - max_val);
        }
        
        size_t target_class = (size_t)targets[b];
        float prob = expf(logits[b * num_classes + target_class] - max_val) / sum_exp;
        float loss = -logf(prob + 1e-8f);
        
        // Atomic Add to average loss across all threads in the batch
        atomicAdd(out_loss, loss / (float)batch_size);
    }
}

extern "C" __global__ void cross_entropy_backward_f32(
    const float* logits,
    const float* targets,
    const float* grad_out,
    float* grad_logits,
    const size_t batch_size,
    const size_t num_classes
) {
    size_t b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b < batch_size) {
        float max_val = -1e20f;
        for (size_t c = 0; c < num_classes; ++c) {
            float val = logits[b * num_classes + c];
            if (val > max_val) max_val = val;
        }
        float sum_exp = 0.0f;
        for (size_t c = 0; c < num_classes; ++c) {
            sum_exp += expf(logits[b * num_classes + c] - max_val);
        }
        
        size_t target_class = (size_t)targets[b];
        float go = grad_out[0] / (float)batch_size;
        
        for (size_t c = 0; c < num_classes; ++c) {
            float prob = expf(logits[b * num_classes + c] - max_val) / sum_exp;
            float g = prob;
            if (c == target_class) g -= 1.0f;
            
            atomicAdd(&grad_logits[b * num_classes + c], g * go);
        }
    }
}

// Output-driven reduction mapping: 1 thread = 1 output element
extern "C" __global__ void sum_f32(
    const float* a, float* out,
    const size_t out_size, const size_t reduced_dim_size, const size_t reduced_dim_stride,
    const size_t out_rank, const size_t os0, const size_t os1, const size_t os2,
    const size_t is0, const size_t is1, const size_t is2
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < out_size) {
        size_t temp = i;
        size_t base_idx = 0;
        
        // Reconstruct base index in input tensor from flat output index
        if (out_rank > 2) { size_t c = temp % os2; temp /= os2; base_idx += c * is2; }
        if (out_rank > 1) { size_t c = temp % os1; temp /= os1; base_idx += c * is1; }
        if (out_rank > 0) { size_t c = temp % os0; temp /= os0; base_idx += c * is0; }

        float sum = 0.0f;
        for (size_t k = 0; k < reduced_dim_size; ++k) {
            sum += a[base_idx + k * reduced_dim_stride];
        }
        out[i] = sum;
    }
}

extern "C" __global__ void sum_backward_f32(
    const float* grad_out, float* grad_a,
    const size_t out_size, const size_t reduced_dim_size, const size_t reduced_dim_stride,
    const size_t out_rank, const size_t os0, const size_t os1, const size_t os2,
    const size_t is0, const size_t is1, const size_t is2
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < out_size) {
        size_t temp = i;
        size_t base_idx = 0;
        if (out_rank > 2) { size_t c = temp % os2; temp /= os2; base_idx += c * is2; }
        if (out_rank > 1) { size_t c = temp % os1; temp /= os1; base_idx += c * is1; }
        if (out_rank > 0) { size_t c = temp % os0; temp /= os0; base_idx += c * is0; }

        float go = grad_out[i];
        for (size_t k = 0; k < reduced_dim_size; ++k) {
            // Broadcast the gradient to all items that were summed
            grad_a[base_idx + k * reduced_dim_stride] += go;
        }
    }
}

extern "C" __global__ void max_f32(
    const float* a, float* out,
    const size_t out_size, const size_t reduced_dim_size, const size_t reduced_dim_stride,
    const size_t out_rank, const size_t os0, const size_t os1, const size_t os2,
    const size_t is0, const size_t is1, const size_t is2
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < out_size) {
        size_t temp = i;
        size_t base_idx = 0;
        if (out_rank > 2) { size_t c = temp % os2; temp /= os2; base_idx += c * is2; }
        if (out_rank > 1) { size_t c = temp % os1; temp /= os1; base_idx += c * is1; }
        if (out_rank > 0) { size_t c = temp % os0; temp /= os0; base_idx += c * is0; }

        float max_val = -1e20f;
        for (size_t k = 0; k < reduced_dim_size; ++k) {
            float val = a[base_idx + k * reduced_dim_stride];
            if (val > max_val) max_val = val;
        }
        out[i] = max_val;
    }
}

extern "C" __global__ void max_backward_f32(
    const float* a, const float* grad_out, float* grad_a,
    const size_t out_size, const size_t reduced_dim_size, const size_t reduced_dim_stride,
    const size_t out_rank, const size_t os0, const size_t os1, const size_t os2,
    const size_t is0, const size_t is1, const size_t is2
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < out_size) {
        size_t temp = i;
        size_t base_idx = 0;
        if (out_rank > 2) { size_t c = temp % os2; temp /= os2; base_idx += c * is2; }
        if (out_rank > 1) { size_t c = temp % os1; temp /= os1; base_idx += c * is1; }
        if (out_rank > 0) { size_t c = temp % os0; temp /= os0; base_idx += c * is0; }

        // Recompute Argmax to avoid storing it in VRAM during forward pass
        float max_val = -1e20f;
        size_t best_k = 0;
        for (size_t k = 0; k < reduced_dim_size; ++k) {
            float val = a[base_idx + k * reduced_dim_stride];
            if (val > max_val) { 
                max_val = val; 
                best_k = k; 
            }
        }
        
        // Route gradient ONLY to the winning index
        grad_a[base_idx + best_k * reduced_dim_stride] += grad_out[i];
    }
}

extern "C" __global__ void bmm_f32(
    const float* a, const float* b, float* out,
    const size_t batch, const size_t m, const size_t k, const size_t n,
    const bool trans_b
) {
    size_t col = blockIdx.x * blockDim.x + threadIdx.x; // 0 to n
    size_t row = blockIdx.y * blockDim.y + threadIdx.y; // 0 to m
    size_t b_idx = blockIdx.z * blockDim.z + threadIdx.z; // 0 to batch

    if (b_idx < batch && row < m && col < n) {
        float sum = 0.0f;
        const float* a_batch = a + b_idx * (m * k);
        const float* b_batch = b + b_idx * (trans_b ? (n * k) : (k * n));

        for (size_t i = 0; i < k; ++i) {
            float val_a = a_batch[row * k + i];
            // Memory-free inline transpose!
            float val_b = trans_b ? b_batch[col * k + i] : b_batch[i * n + col];
            sum += val_a * val_b;
        }
        out[b_idx * (m * n) + row * n + col] = sum;
    }
}

// BMM Backward A (trans_b = false) => dA = dC @ B^T
extern "C" __global__ void bmm_backward_a_f32(
    const float* grad_out, const float* b, float* grad_a,
    const size_t batch, const size_t m, const size_t k, const size_t n
) {
    size_t col = blockIdx.x * blockDim.x + threadIdx.x; // 0 to k
    size_t row = blockIdx.y * blockDim.y + threadIdx.y; // 0 to m
    size_t b_idx = blockIdx.z * blockDim.z + threadIdx.z;

    if (b_idx < batch && row < m && col < k) {
        float sum = 0.0f;
        const float* b_batch = b + b_idx * (k * n);
        const float* go_batch = grad_out + b_idx * (m * n);
        for (size_t i = 0; i < n; ++i) { sum += go_batch[row * n + i] * b_batch[col * n + i]; }
        atomicAdd(&grad_a[b_idx * (m * k) + row * k + col], sum);
    }
}

// BMM Backward B (trans_b = false) => dB = A^T @ dC
extern "C" __global__ void bmm_backward_b_f32(
    const float* a, const float* grad_out, float* grad_b,
    const size_t batch, const size_t m, const size_t k, const size_t n
) {
    size_t col = blockIdx.x * blockDim.x + threadIdx.x; // 0 to n
    size_t row = blockIdx.y * blockDim.y + threadIdx.y; // 0 to k
    size_t b_idx = blockIdx.z * blockDim.z + threadIdx.z;

    if (b_idx < batch && row < k && col < n) {
        float sum = 0.0f;
        const float* a_batch = a + b_idx * (m * k);
        const float* go_batch = grad_out + b_idx * (m * n);
        for (size_t i = 0; i < m; ++i) { sum += a_batch[i * k + row] * go_batch[i * n + col]; }
        atomicAdd(&grad_b[b_idx * (k * n) + row * n + col], sum);
    }
}

// BMM Backward A (trans_b = true) => dA = dC @ B
extern "C" __global__ void bmm_backward_a_transb_f32(
    const float* grad_out, const float* b, float* grad_a,
    const size_t batch, const size_t m, const size_t k, const size_t n
) {
    size_t col = blockIdx.x * blockDim.x + threadIdx.x; // 0 to k
    size_t row = blockIdx.y * blockDim.y + threadIdx.y; // 0 to m
    size_t b_idx = blockIdx.z * blockDim.z + threadIdx.z;

    if (b_idx < batch && row < m && col < k) {
        float sum = 0.0f;
        const float* b_batch = b + b_idx * (n * k);
        const float* go_batch = grad_out + b_idx * (m * n);
        for (size_t i = 0; i < n; ++i) { sum += go_batch[row * n + i] * b_batch[i * k + col]; }
        atomicAdd(&grad_a[b_idx * (m * k) + row * k + col], sum);
    }
}

// BMM Backward B (trans_b = true) => dB = dC^T @ A
extern "C" __global__ void bmm_backward_b_transb_f32(
    const float* a, const float* grad_out, float* grad_b,
    const size_t batch, const size_t m, const size_t k, const size_t n
) {
    size_t col = blockIdx.x * blockDim.x + threadIdx.x; // 0 to k
    size_t row = blockIdx.y * blockDim.y + threadIdx.y; // 0 to n
    size_t b_idx = blockIdx.z * blockDim.z + threadIdx.z;

    if (b_idx < batch && row < n && col < k) {
        float sum = 0.0f;
        const float* a_batch = a + b_idx * (m * k);
        const float* go_batch = grad_out + b_idx * (m * n);
        for (size_t i = 0; i < m; ++i) { sum += go_batch[i * n + row] * a_batch[i * k + col]; }
        atomicAdd(&grad_b[b_idx * (n * k) + row * k + col], sum);
    }
}

// --- STANDALONE SOFTMAX ---
extern "C" __global__ void softmax_f32(const float* x, float* out, const size_t B, const size_t N) {
    size_t b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b < B) {
        float max_val = -1e20f;
        for (size_t i = 0; i < N; ++i) {
            if (x[b * N + i] > max_val) max_val = x[b * N + i];
        }
        float sum = 0.0f;
        for (size_t i = 0; i < N; ++i) {
            float e = expf(x[b * N + i] - max_val);
            out[b * N + i] = e;
            sum += e;
        }
        for (size_t i = 0; i < N; ++i) { out[b * N + i] /= sum; }
    }
}

extern "C" __global__ void softmax_backward_f32(const float* out, const float* grad_out, float* grad_x, const size_t B, const size_t N) {
    size_t b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b < B) {
        float sum_go = 0.0f;
        for (size_t i = 0; i < N; ++i) { sum_go += out[b * N + i] * grad_out[b * N + i]; }
        for (size_t i = 0; i < N; ++i) {
            float g = out[b * N + i] * (grad_out[b * N + i] - sum_go);
            atomicAdd(&grad_x[b * N + i], g);
        }
    }
}

// --- SPECIALIZED 4D TRANSPOSE FOR MULTI-HEAD ATTENTION ---
// Swaps dimension 1 (Seq) and 2 (Heads): [B, S, H, D] -> [B, H, S, D]
extern "C" __global__ void transpose_0213_f32(
    const float* src, float* dst,
    const size_t B, const size_t S, const size_t H, const size_t D
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = B * S * H * D;
    if (i < total) {
        size_t d = i % D;
        size_t temp = i / D;
        size_t h = temp % H;
        temp /= H;
        size_t s = temp % S;
        size_t b = temp / S;
        
        // Output mapped to [B, H, S, D]
        size_t dst_idx = b * (H * S * D) + h * (S * D) + s * D + d;
        dst[dst_idx] = src[i];
    }
}

// Exactly the reverse mapping. No atomics needed since it's a 1:1 map!
extern "C" __global__ void transpose_0213_backward_f32(
    const float* grad_out, float* grad_src,
    const size_t B, const size_t S, const size_t H, const size_t D
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = B * S * H * D;
    if (i < total) {
        size_t d = i % D;
        size_t temp = i / D;
        size_t h = temp % H;
        temp /= H;
        size_t s = temp % S;
        size_t b = temp / S;
        
        size_t dst_idx = b * (H * S * D) + h * (S * D) + s * D + d;
        grad_src[i] += grad_out[dst_idx]; 
    }
}

// --- ROTARY POSITIONAL EMBEDDINGS (RoPE) ---
// Operates on adjacent pairs [2i, 2i+1] for caching efficiency (NeoX/LLaMA style)
extern "C" __global__ void rope_f32(
    const float* x, float* out,
    const size_t seq_len, const size_t hidden_dim, const size_t head_dim, const size_t num_pairs
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_pairs) {
        size_t d_pair = i % (hidden_dim / 2);
        size_t d_head_pair = d_pair % (head_dim / 2);
        size_t seq_idx = (i / (hidden_dim / 2)) % seq_len;

        size_t idx1 = i * 2;
        size_t idx2 = i * 2 + 1;

        // Compute frequency on the fly to save VRAM: 10000^(-2i/d)
        float freq = powf(10000.0f, -2.0f * (float)d_head_pair / (float)head_dim);
        float angle = (float)seq_idx * freq;
        
        // Fast intrinsic trig functions
        float cos_a = cosf(angle);
        float sin_a = sinf(angle);

        float x1 = x[idx1];
        float x2 = x[idx2];

        // 2D Rotation
        out[idx1] = x1 * cos_a - x2 * sin_a;
        out[idx2] = x2 * cos_a + x1 * sin_a;
    }
}

extern "C" __global__ void rope_backward_f32(
    const float* grad_out, float* grad_x,
    const size_t seq_len, const size_t hidden_dim, const size_t head_dim, const size_t num_pairs
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_pairs) {
        size_t d_pair = i % (hidden_dim / 2);
        size_t d_head_pair = d_pair % (head_dim / 2);
        size_t seq_idx = (i / (hidden_dim / 2)) % seq_len;

        size_t idx1 = i * 2;
        size_t idx2 = i * 2 + 1;

        float freq = powf(10000.0f, -2.0f * (float)d_head_pair / (float)head_dim);
        float angle = (float)seq_idx * freq;
        float cos_a = cosf(angle);
        float sin_a = sinf(angle);

        float g1 = grad_out[idx1];
        float g2 = grad_out[idx2];

        // Inverse Rotation: cos(-a) = cos(a), sin(-a) = -sin(a)
        grad_x[idx1] += g1 * cos_a + g2 * sin_a;
        grad_x[idx2] += g2 * cos_a - g1 * sin_a;
    }
}
