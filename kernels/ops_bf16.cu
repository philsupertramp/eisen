#include "common.cuh"


extern "C" __global__ void cast_f32_to_bf16(const float* src, __nv_bfloat16* dst, const size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = __float2bfloat16(src[i]);
    }
}

extern "C" __global__ void cast_bf16_to_f32(const __nv_bfloat16* src, float* dst, const size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = __bfloat162float(src[i]);
    }
}

extern "C" __global__ void cast_bf16_to_f32_accumulate(const __nv_bfloat16* src, float* dst, const size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        atomicAdd(&dst[i], __bfloat162float(src[i]));
    }
}

extern "C" __global__ void gather_bf16_f32(
    const __nv_bfloat16* weights,
    const float* indices,
    float* out,
    const size_t hidden_dim,
    const size_t out_size
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < out_size) {
        size_t row = i / hidden_dim;
        size_t col = i % hidden_dim;
        size_t w_row = (size_t)indices[row];
        out[i] = __bfloat162float(weights[w_row * hidden_dim + col]);
    }
}

extern "C" __global__ void rmsnorm_f32_bf16w(
    const float* x, const __nv_bfloat16* w, float* out,
    const size_t dim, const float eps, const size_t num_vecs
) {
    size_t n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n < num_vecs) {
        size_t offset = n * dim;
        float sum_sq = 0.0f;
        for (size_t d = 0; d < dim; ++d) { float val = bf16q(x[offset + d]); sum_sq += bf16q(val * val); }
        float rrms = rsqrtf(sum_sq / (float)dim + eps);
        for (size_t d = 0; d < dim; ++d) {
            out[offset + d] = bf16q(bf16q(x[offset + d]) * bf16q(rrms) * __bfloat162float(w[d]));
        }
    }
}

extern "C" __global__ void rmsnorm_backward_bf16w_f32(
    const float* x, const __nv_bfloat16* w, const float* grad_out,
    float* grad_x, float* grad_w,
    const size_t dim, const float eps, const size_t num_vecs
) {
    size_t n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n < num_vecs) {
        size_t offset = n * dim;
        float sum_sq = 0.0f;
        for (size_t d = 0; d < dim; ++d) { float val = bf16q(x[offset + d]); sum_sq += bf16q(val * val); }
        float rrms = rsqrtf(sum_sq / (float)dim + eps);
        float grad_dot_x_w = 0.0f;
        for (size_t d = 0; d < dim; ++d) {
            float wv = __bfloat162float(w[d]);
            grad_dot_x_w += bf16q(grad_out[offset + d]) * bf16q(x[offset + d]) * bf16q(wv);
        }
        float rrc_d = (rrms * rrms * rrms) / (float)dim;
        for (size_t d = 0; d < dim; ++d) {
            float val_x = bf16q(x[offset + d]);
            float val_w = __bfloat162float(w[d]);
            float go = bf16q(grad_out[offset + d]);
            float dx = bf16q(rrms * bf16q(go * val_w) - bf16q(val_x * rrc_d * grad_dot_x_w));
            grad_x[offset + d] += dx;
            atomicAdd(&grad_w[d], bf16q(go * val_x * rrms));
        }
    }
}



extern "C" __global__ void adamw_step_bf16mom_f32(
    float* weights,
    const float* grads,
    __nv_bfloat16* m,
    __nv_bfloat16* v,
    const float lr,
    const float beta1,
    const float beta2,
    const float eps,
    const float weight_decay,
    const float bc1,
    const float bc2,
    const size_t n
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float g = grads[i];
        float w = weights[i];

        w *= (1.0f - lr * weight_decay);

        float m_old = __bfloat162float(m[i]);
        float v_old = __bfloat162float(v[i]);
        float m_new = beta1 * m_old + (1.0f - beta1) * g;
        float v_new = beta2 * v_old + (1.0f - beta2) * g * g;

        m[i] = __float2bfloat16(m_new);
        v[i] = __float2bfloat16(v_new);

        float m_hat = m_new * bc1;
        float v_hat = v_new * bc2;
        weights[i] = w - lr * m_hat / (sqrtf(v_hat) + eps);
    }
}

extern "C" __global__ void adamw_step_bf16mom(
    float* weights,
    const __nv_bfloat16* grads,
    __nv_bfloat16* m,
    __nv_bfloat16* v,
    const float lr,
    const float beta1,
    const float beta2,
    const float eps,
    const float weight_decay,
    const float bc1,
    const float bc2,
    const size_t n
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float g = __bfloat162float(grads[i]);
        float w = weights[i];

        w *= (1.0f - lr * weight_decay);

        float m_old = __bfloat162float(m[i]);
        float v_old = __bfloat162float(v[i]);
        float m_new = beta1 * m_old + (1.0f - beta1) * g;
        float v_new = beta2 * v_old + (1.0f - beta2) * g * g;

        m[i] = __float2bfloat16(m_new);
        v[i] = __float2bfloat16(v_new);

        float m_hat = m_new * bc1;
        float v_hat = v_new * bc2;
        weights[i] = w - lr * m_hat / (sqrtf(v_hat) + eps);
    }
}

extern "C" __global__ void adamw_step_bf16w_bf16mom_f32(
    __nv_bfloat16* weights,
    const float* grads,
    __nv_bfloat16* m,
    __nv_bfloat16* v,
    const float lr,
    const float beta1,
    const float beta2,
    const float eps,
    const float weight_decay,
    const float bc1,
    const float bc2,
    const size_t n
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float g = grads[i];
        float w = __bfloat162float(weights[i]);

        w *= (1.0f - lr * weight_decay);

        float m_old = __bfloat162float(m[i]);
        float v_old = __bfloat162float(v[i]);
        float m_new = beta1 * m_old + (1.0f - beta1) * g;
        float v_new = beta2 * v_old + (1.0f - beta2) * g * g;

        m[i] = __float2bfloat16(m_new);
        v[i] = __float2bfloat16(v_new);

        float m_hat = m_new * bc1;
        float v_hat = v_new * bc2;
        float w_new = w - lr * m_hat / (sqrtf(v_hat) + eps);
        weights[i] = __float2bfloat16(w_new);
    }
}

// ── ADD ──────────────────────────────────────────────────────────────────────

// BF16 inputs + BF16 output (broadcast-aware)
extern "C" __global__ void add_bf16(
    const __nv_bfloat16* a,
    const __nv_bfloat16* b,
    __nv_bfloat16* out,
    const size_t n,
    const size_t rank,
    const size_t s0, const size_t s1, const size_t s2,
    const size_t a0, const size_t a1, const size_t a2,
    const size_t b0, const size_t b1, const size_t b2
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        size_t temp = i, idx_a = 0, idx_b = 0;
        if (rank > 2) { size_t c = temp % s2; temp /= s2; idx_a += c * a2; idx_b += c * b2; }
        if (rank > 1) { size_t c = temp % s1; temp /= s1; idx_a += c * a1; idx_b += c * b1; }
        if (rank > 0) { size_t c = temp % s0; temp /= s0; idx_a += c * a0; idx_b += c * b0; }
        out[i] = __float2bfloat16(
            __bfloat162float(a[idx_a]) + __bfloat162float(b[idx_b])
        );
    }
}

// BF16 output, mixed: LHS BF16, RHS FP32 (e.g. residual + mask)
extern "C" __global__ void add_bf16lhs_f32rhs_bf16out(
    const __nv_bfloat16* a,
    const float*          b,
    __nv_bfloat16*        out,
    const size_t n,
    const size_t rank,
    const size_t s0, const size_t s1, const size_t s2,
    const size_t a0, const size_t a1, const size_t a2,
    const size_t b0, const size_t b1, const size_t b2
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        size_t temp = i, idx_a = 0, idx_b = 0;
        if (rank > 2) { size_t c = temp % s2; temp /= s2; idx_a += c * a2; idx_b += c * b2; }
        if (rank > 1) { size_t c = temp % s1; temp /= s1; idx_a += c * a1; idx_b += c * b1; }
        if (rank > 0) { size_t c = temp % s0; temp /= s0; idx_a += c * a0; idx_b += c * b0; }
        out[i] = __float2bfloat16(
            __bfloat162float(a[idx_a]) + b[idx_b]
        );
    }
}

// Backward accumulation into BF16 grad buffer
// grad_out is FP32 (parameter grads stay FP32); grad_target is BF16 (activation grad)
extern "C" __global__ void accumulate_bf16out(
    __nv_bfloat16*  grad_target,
    const float*    grad_out,
    const size_t n,
    const size_t rank,
    const size_t s0, const size_t s1, const size_t s2,
    const size_t t0, const size_t t1, const size_t t2
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        size_t temp = i, idx_t = 0;
        if (rank > 2) { size_t c = temp % s2; temp /= s2; idx_t += c * t2; }
        if (rank > 1) { size_t c = temp % s1; temp /= s1; idx_t += c * t1; }
        if (rank > 0) { size_t c = temp % s0; temp /= s0; idx_t += c * t0; }
        // atomic BF16 accumulate via FP32 round-trip (no native BF16 atomicAdd on most hw)
        float old_f = __bfloat162float(grad_target[idx_t]);
        grad_target[idx_t] = __float2bfloat16(old_f + grad_out[i]);
    }
}

// Pure BF16 accumulate for Reshape/View backward
extern "C" __global__ void accumulate_bf16(
    __nv_bfloat16* grad_target,
    const __nv_bfloat16* grad_out,
    const size_t n,
    const size_t rank,
    const size_t s0, const size_t s1, const size_t s2,
    const size_t t0, const size_t t1, const size_t t2
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        size_t temp = i, idx_t = 0;
        if (rank > 2) { size_t c = temp % s2; temp /= s2; idx_t += c * t2; }
        if (rank > 1) { size_t c = temp % s1; temp /= s1; idx_t += c * t1; }
        if (rank > 0) { size_t c = temp % s0; temp /= s0; idx_t += c * t0; }

        // Local FP32 round-trip for safe gradient accumulation
        float old_f = __bfloat162float(grad_target[idx_t]);
        float incoming = __bfloat162float(grad_out[i]);
        grad_target[idx_t] = __float2bfloat16(old_f + incoming);
    }
}

// Transpose 0213 backward (reading BF16 grad_out -> accumulating BF16 grad_src)
extern "C" __global__ void transpose_0213_backward_bf16(
    const __nv_bfloat16* grad_out,
    __nv_bfloat16* grad_src,
    const size_t B, const size_t S, const size_t H, const size_t D
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = B * S * H * D;
    if (i < total) {
        size_t d = i % D; size_t tmp = i / D;
        size_t h = tmp % H; tmp /= H;
        size_t s = tmp % S; size_t b = tmp / S;
        size_t dst_idx = b*(H*S*D) + h*(S*D) + s*D + d;

        // Local FP32 accumulation
        float current = __bfloat162float(grad_src[i]);
        float incoming = __bfloat162float(grad_out[dst_idx]);
        grad_src[i] = __float2bfloat16(current + incoming);
    }
}
// ── MUL ──────────────────────────────────────────────────────────────────────

extern "C" __global__ void mul_bf16(
    const __nv_bfloat16* a, const __nv_bfloat16* b,
    __nv_bfloat16* out, const size_t n
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = __float2bfloat16(
            __bfloat162float(a[i]) * __bfloat162float(b[i])
        );
}

// Mixed: BF16 activation * FP32 scalar tensor -> BF16 out (e.g. attention scale)
extern "C" __global__ void mul_bf16lhs_f32rhs_bf16out(
    const __nv_bfloat16* a, const float* b,
    __nv_bfloat16* out, const size_t n
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = __float2bfloat16(__bfloat162float(a[i]) * b[i]);
}

extern "C" __global__ void mul_backward_bf16(
    const __nv_bfloat16* a, const __nv_bfloat16* b, const __nv_bfloat16* grad_out,
    __nv_bfloat16* grad_a, __nv_bfloat16* grad_b, const size_t n
) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        float f_a = __bfloat162float(a[i]);
        float f_b = __bfloat162float(b[i]);
        float f_go = __bfloat162float(grad_out[i]);

        float current_ga = __bfloat162float(grad_a[i]);
        float current_gb = __bfloat162float(grad_b[i]);

        grad_a[i] = __float2bfloat16(current_ga + (f_b * f_go));
        grad_b[i] = __float2bfloat16(current_gb + (f_a * f_go));
    }
}

// Backward: BF16 a, BF16 b, FP32 grad_out -> FP32 grad_a, FP32 grad_b
extern "C" __global__ void mul_backward_bf16in_f32(
    const __nv_bfloat16* a, const __nv_bfloat16* b,
    const float* grad_out,
    float* grad_a, float* grad_b,
    const size_t n
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float av = __bfloat162float(a[i]);
        float bv = __bfloat162float(b[i]);
        float go = grad_out[i];
        grad_a[i] += bv * go;
        grad_b[i] += av * go;
    }
}

// Mixed backward: BF16 a, FP32 b (e.g. scale), FP32 grad -> FP32 grads
extern "C" __global__ void mul_backward_bf16lhs_f32rhs(
    const __nv_bfloat16* a, const float* b,
    const float* grad_out,
    float* grad_a, float* grad_b,
    const size_t n
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float av = __bfloat162float(a[i]);
        float go = grad_out[i];
        grad_a[i] += b[i] * go;
        grad_b[i] += av  * go;
    }
}

// ── SILU ─────────────────────────────────────────────────────────────────────

// BF16 input -> BF16 output
extern "C" __global__ void silu_bf16(
    const __nv_bfloat16* x, __nv_bfloat16* out, const size_t n
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float val = __bfloat162float(x[i]);
        out[i] = __float2bfloat16(val / (1.0f + expf(-val)));
    }
}

// Backward: BF16 x, FP32 grad_out -> FP32 grad_x
extern "C" __global__ void silu_backward_bf16in_f32(
    const __nv_bfloat16* x, const float* grad_out,
    float* grad_x, const size_t n
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float val  = __bfloat162float(x[i]);
        float sig  = 1.0f / (1.0f + expf(-val));
        float silu = val * sig;
        float d    = silu + sig * (1.0f - silu);
        grad_x[i] += grad_out[i] * d;
    }
}

extern "C" __global__ void silu_backward_bf16(
    const __nv_bfloat16* x, const __nv_bfloat16* grad_out, 
    __nv_bfloat16* grad_x, const size_t n
) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        float val = __bfloat162float(x[i]);
        float go = __bfloat162float(grad_out[i]);

        float sig = 1.0f / (1.0f + expf(-val));
        float silu = val * sig;
        float d_silu = silu + sig * (1.0f - silu);

        float current_gx = __bfloat162float(grad_x[i]);
        grad_x[i] = __float2bfloat16(current_gx + (go * d_silu));
    }
}

// --- MAX REDUCTION ---
extern "C" __global__ void max_bf16(
    const __nv_bfloat16* a, __nv_bfloat16* out,
    const size_t out_size, const size_t reduced_dim_size, const size_t reduced_dim_stride,
    const size_t out_rank, const size_t os0, const size_t os1, const size_t os2,
    const size_t is0, const size_t is1, const size_t is2
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < out_size) {
        size_t temp = i; size_t base_idx = 0;
        if (out_rank > 2) { size_t c = temp % os2; temp /= os2; base_idx += c * is2; }
        if (out_rank > 1) { size_t c = temp % os1; temp /= os1; base_idx += c * is1; }
        if (out_rank > 0) { size_t c = temp % os0; temp /= os0; base_idx += c * is0; }
        
        float max_val = -1e20f;
        for (size_t k = 0; k < reduced_dim_size; ++k) {
            float val = __bfloat162float(a[base_idx + k * reduced_dim_stride]);
            if (val > max_val) max_val = val;
        }
        out[i] = __float2bfloat16(max_val);
    }
}

extern "C" __global__ void max_backward_bf16(
    const __nv_bfloat16* a, const __nv_bfloat16* grad_out, __nv_bfloat16* grad_a,
    const size_t out_size, const size_t reduced_dim_size, const size_t reduced_dim_stride,
    const size_t out_rank, const size_t os0, const size_t os1, const size_t os2,
    const size_t is0, const size_t is1, const size_t is2
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < out_size) {
        size_t temp = i; size_t base_idx = 0;
        if (out_rank > 2) { size_t c = temp % os2; temp /= os2; base_idx += c * is2; }
        if (out_rank > 1) { size_t c = temp % os1; temp /= os1; base_idx += c * is1; }
        if (out_rank > 0) { size_t c = temp % os0; temp /= os0; base_idx += c * is0; }
        
        float max_val = -1e20f; size_t best_k = 0;
        for (size_t k = 0; k < reduced_dim_size; ++k) {
            float val = __bfloat162float(a[base_idx + k * reduced_dim_stride]);
            if (val > max_val) { max_val = val; best_k = k; }
        }
        
        size_t idx = base_idx + best_k * reduced_dim_stride;
        float go = __bfloat162float(grad_out[i]);
        float current_ga = __bfloat162float(grad_a[idx]);
        grad_a[idx] = __float2bfloat16(current_ga + go);
    }
}

// ── SOFTMAX ──────────────────────────────────────────────────────────────────

// BF16 input -> BF16 output (FP32 accumulation internally)
extern "C" __global__ void softmax_bf16(
    const __nv_bfloat16* x, __nv_bfloat16* out,
    const size_t B, const size_t N
) {
    size_t b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b < B) {
        float max_val = -1e20f;
        for (size_t i = 0; i < N; ++i) {
            float v = __bfloat162float(x[b * N + i]);
            if (v > max_val) max_val = v;
        }
        float sum = 0.0f;
        for (size_t i = 0; i < N; ++i)
            sum += expf(__bfloat162float(x[b * N + i]) - max_val);
        for (size_t i = 0; i < N; ++i) {
            float e = expf(__bfloat162float(x[b * N + i]) - max_val);
            out[b * N + i] = __float2bfloat16(e / sum);
        }
    }
}
extern "C" __global__ void softmax_backward_bf16(
    const __nv_bfloat16* out, const __nv_bfloat16* grad_out,
    __nv_bfloat16* grad_x, const size_t B, const size_t N
) {
    size_t b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b < B) {
        // 1. Accumulate the dot product entirely in FP32 registers
        float sum_go = 0.0f;
        for (size_t i = 0; i < N; ++i) {
            float ov = __bfloat162float(out[b * N + i]);
            float go = __bfloat162float(grad_out[b * N + i]);
            sum_go += ov * go;
        }

        // 2. Compute the local gradient and cast down for VRAM storage
        for (size_t i = 0; i < N; ++i) {
            float ov = __bfloat162float(out[b * N + i]);
            float go = __bfloat162float(grad_out[b * N + i]);
            
            // Derivative math stays in FP32
            float dx = ov * (go - sum_go);

            // Emulate the accumulation by reading the existing BF16 value, 
            // adding in FP32, and writing back.
            float current_grad = __bfloat162float(grad_x[b * N + i]);
            grad_x[b * N + i] = __float2bfloat16(current_grad + dx);
        }
    }
}
// Backward: BF16 softmax output, FP32 grad_out -> FP32 grad_x
extern "C" __global__ void softmax_backward_bf16in_f32(
    const __nv_bfloat16* out, const float* grad_out,
    float* grad_x, const size_t B, const size_t N
) {
    size_t b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b < B) {
        float sum_go = 0.0f;
        for (size_t i = 0; i < N; ++i)
            sum_go += __bfloat162float(out[b * N + i]) * grad_out[b * N + i];
        for (size_t i = 0; i < N; ++i) {
            float ov = __bfloat162float(out[b * N + i]);
            atomicAdd(&grad_x[b * N + i], ov * (grad_out[b * N + i] - sum_go));
        }
    }
}

// --- SUM REDUCTION ---
extern "C" __global__ void sum_bf16(
    const __nv_bfloat16* a, __nv_bfloat16* out,
    const size_t out_size, const size_t reduced_dim_size, const size_t reduced_dim_stride,
    const size_t out_rank, const size_t os0, const size_t os1, const size_t os2,
    const size_t is0, const size_t is1, const size_t is2
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < out_size) {
        size_t temp = i; size_t base_idx = 0;
        if (out_rank > 2) { size_t c = temp % os2; temp /= os2; base_idx += c * is2; }
        if (out_rank > 1) { size_t c = temp % os1; temp /= os1; base_idx += c * is1; }
        if (out_rank > 0) { size_t c = temp % os0; temp /= os0; base_idx += c * is0; }
        
        float sum = 0.0f;
        for (size_t k = 0; k < reduced_dim_size; ++k) {
            sum += __bfloat162float(a[base_idx + k * reduced_dim_stride]);
        }
        out[i] = __float2bfloat16(sum);
    }
}

extern "C" __global__ void sum_backward_bf16(
    const __nv_bfloat16* grad_out, __nv_bfloat16* grad_a,
    const size_t out_size, const size_t reduced_dim_size, const size_t reduced_dim_stride,
    const size_t out_rank, const size_t os0, const size_t os1, const size_t os2,
    const size_t is0, const size_t is1, const size_t is2
) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < out_size; i += blockDim.x * gridDim.x) {
        size_t temp = i; size_t base_idx = 0;
        if (out_rank > 2) { size_t c = temp % os2; temp /= os2; base_idx += c * is2; }
        if (out_rank > 1) { size_t c = temp % os1; temp /= os1; base_idx += c * is1; }
        if (out_rank > 0) { size_t c = temp % os0; temp /= os0; base_idx += c * is0; }
        
        float go = __bfloat162float(grad_out[i]);
        for (size_t k = 0; k < reduced_dim_size; ++k) {
            size_t idx = base_idx + k * reduced_dim_stride;
            float current_ga = __bfloat162float(grad_a[idx]);
            grad_a[idx] = __float2bfloat16(current_ga + go);
        }
    }
}


// ── COPY (RESHAPE) ───────────────────────────────────────────────────────────

extern "C" __global__ void copy_bf16(
    const __nv_bfloat16* src, __nv_bfloat16* dst, const size_t n
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i];
}

// copy_bf16 backward: FP32 grad flows back unchanged — use accumulate_f32.
// No new kernel needed: grad buffers are FP32.

// ── TRANSPOSE 0213 ───────────────────────────────────────────────────────────

extern "C" __global__ void transpose_0213_bf16(
    const __nv_bfloat16* src, __nv_bfloat16* dst,
    const size_t B, const size_t S, const size_t H, const size_t D
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = B * S * H * D;
    if (i < total) {
        size_t d = i % D; size_t tmp = i / D;
        size_t h = tmp % H; tmp /= H;
        size_t s = tmp % S; size_t b = tmp / S;
        size_t dst_idx = b*(H*S*D) + h*(S*D) + s*D + d;
        dst[dst_idx] = src[i];
    }
}

// Backward is identical permutation (self-inverse transpose),
// reading FP32 grad_out -> FP32 grad_src — same as transpose_0213_backward_f32.
// No new kernel needed.

// ── ROPE ─────────────────────────────────────────────────────────────────────

// BF16 input -> BF16 output
extern "C" __global__ void rope_bf16(
    const __nv_bfloat16* x, __nv_bfloat16* out,
    const size_t seq_len, const size_t hidden_dim,
    const size_t head_dim, const size_t num_pairs
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_pairs) {
        size_t d_pair      = i % (hidden_dim / 2);
        size_t d_head_pair = d_pair % (head_dim / 2);
        size_t seq_idx     = (i / (hidden_dim / 2)) % seq_len;
        size_t idx1 = i * 2, idx2 = i * 2 + 1;
        float freq  = powf(10000.0f, -2.0f * (float)d_head_pair / (float)head_dim);
        float angle = (float)seq_idx * freq;
        float cos_a = cosf(angle), sin_a = sinf(angle);
        float x1 = __bfloat162float(x[idx1]);
        float x2 = __bfloat162float(x[idx2]);
        out[idx1] = __float2bfloat16(x1 * cos_a - x2 * sin_a);
        out[idx2] = __float2bfloat16(x2 * cos_a + x1 * sin_a);
    }
}
// Pure BF16 RoPE backward (recomputes inverse rotation)
extern "C" __global__ void rope_backward_bf16(
    const __nv_bfloat16* grad_out,
    __nv_bfloat16* grad_x,
    const size_t seq_len, const size_t hidden_dim,
    const size_t head_dim, const size_t num_pairs
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_pairs) {
        size_t d_pair      = i % (hidden_dim / 2);
        size_t d_head_pair = d_pair % (head_dim / 2);
        size_t seq_idx     = (i / (hidden_dim / 2)) % seq_len;
        
        size_t idx1 = i * 2;
        size_t idx2 = i * 2 + 1;

        // Recompute position frequencies
        float freq  = powf(10000.0f, -2.0f * (float)d_head_pair / (float)head_dim);
        float angle = (float)seq_idx * freq;
        float cos_a = cosf(angle);
        float sin_a = sinf(angle);

        // Read incoming gradients into FP32 registers
        float go1 = __bfloat162float(grad_out[idx1]);
        float go2 = __bfloat162float(grad_out[idx2]);

        // Inverse rotation: applying -angle
        float dx1 = go1 * cos_a + go2 * sin_a;
        float dx2 = go2 * cos_a - go1 * sin_a;

        // Local FP32 accumulation into the existing gradient buffer
        float curr_gx1 = __bfloat162float(grad_x[idx1]);
        float curr_gx2 = __bfloat162float(grad_x[idx2]);

        grad_x[idx1] = __float2bfloat16(curr_gx1 + dx1);
        grad_x[idx2] = __float2bfloat16(curr_gx2 + dx2);
    }
}
// RoPE backward: recomputes cos/sin from position — no saved activation needed.
// FP32 grad_out -> FP32 grad_x. Use rope_backward_f32 unchanged.

// ── RMSNORM ──────────────────────────────────────────────────────────────────

// BF16 x, BF16 w -> BF16 out  (accumulates in FP32 internally)
extern "C" __global__ void rmsnorm_bf16(
    const __nv_bfloat16* x, const __nv_bfloat16* w,
    __nv_bfloat16* out,
    const size_t dim, const float eps, const size_t num_vecs
) {
    size_t n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n < num_vecs) {
        size_t off = n * dim;
        float sum_sq = 0.0f;
        for (size_t d = 0; d < dim; ++d) {
            float v = __bfloat162float(x[off + d]);
            sum_sq += v * v;
        }
        float rrms = rsqrtf(sum_sq / (float)dim + eps);
        for (size_t d = 0; d < dim; ++d) {
            float xv = __bfloat162float(x[off + d]);
            float wv = __bfloat162float(w[d]);
            out[off + d] = __float2bfloat16(xv * rrms * wv);
        }
    }
}

extern "C" __global__ void fill_bf16(__nv_bfloat16* data, const __nv_bfloat16 value, const size_t n) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        data[i] = value;
    }
}


// Helper function for thread-safe BF16 accumulation without hardware atomics
__device__ inline void atomicAdd_bf16(__nv_bfloat16* address, float val) {
    unsigned short* address_as_us = (unsigned short*)address;
    unsigned short old = *address_as_us, assumed;
    do {
        assumed = old;
        // Unpack the current value to f32, add the gradient, and repack to bf16
        float current_val = __bfloat162float(*(__nv_bfloat16*)&assumed);
        float new_val = current_val + val;
        unsigned short new_val_us = *(unsigned short*)&(__float2bfloat16(new_val));
        
        // Attempt to swap. If another thread modified it, loop and try again.
        old = atomicCAS(address_as_us, assumed, new_val_us);
    } while (assumed != old);
}

extern "C" __global__ void rmsnorm_backward_bf16(
    const __nv_bfloat16* x, const __nv_bfloat16* w, const __nv_bfloat16* grad_out,
    __nv_bfloat16* grad_x, __nv_bfloat16* grad_w,
    const size_t dim, const float eps, const size_t num_vecs
) {
    size_t n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n < num_vecs) {
        size_t offset = n * dim;
        
        // 1. Calculate sum of squares (in f32 registers)
        float sum_sq = 0.0f;
        for (size_t d = 0; d < dim; ++d) { 
            float val = __bfloat162float(x[offset + d]); 
            sum_sq += val * val; 
        }
        float rrms = rsqrtf(sum_sq / (float)dim + eps);
        
        // 2. Calculate local dot product of grad_out * x * w (in f32 registers)
        float grad_dot_x_w = 0.0f;
        for (size_t d = 0; d < dim; ++d) {
            float go = __bfloat162float(grad_out[offset + d]);
            float val_x = __bfloat162float(x[offset + d]);
            float val_w = __bfloat162float(w[d]);
            grad_dot_x_w += go * val_x * val_w;
        }
        
        float rrc_d = (rrms * rrms * rrms) / (float)dim;
        
        // 3. Compute final gradients and write to VRAM
        for (size_t d = 0; d < dim; ++d) {
            float val_x = __bfloat162float(x[offset + d]);
            float val_w = __bfloat162float(w[d]);
            float go = __bfloat162float(grad_out[offset + d]);
            
            // grad_x: Unique to this thread, so standard read-add-write is safe
            float dx = rrms * (go * val_w) - (val_x * rrc_d * grad_dot_x_w);
            float current_gx = __bfloat162float(grad_x[offset + d]);
            grad_x[offset + d] = __float2bfloat16(current_gx + dx);
            
            // grad_w: Shared across all threads. Must use software atomicCAS.
            float dw = go * val_x * rrms;
            atomicAdd_bf16(&grad_w[d], dw);
        }
    }
}
// Backward: BF16 x, BF16 w, FP32 grad_out -> FP32 grad_x, FP32 grad_w
extern "C" __global__ void rmsnorm_backward_bf16in_f32(
    const __nv_bfloat16* x, const __nv_bfloat16* w,
    const float* grad_out,
    float* grad_x, float* grad_w,
    const size_t dim, const float eps, const size_t num_vecs
) {
    size_t n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n < num_vecs) {
        size_t off = n * dim;
        float sum_sq = 0.0f;
        for (size_t d = 0; d < dim; ++d) {
            float v = __bfloat162float(x[off + d]);
            sum_sq += v * v;
        }
        float rrms = rsqrtf(sum_sq / (float)dim + eps);
        float gdxw = 0.0f;
        for (size_t d = 0; d < dim; ++d)
            gdxw += grad_out[off + d] * __bfloat162float(x[off + d]) * __bfloat162float(w[d]);
        float rrc_d = (rrms * rrms * rrms) / (float)dim;
        for (size_t d = 0; d < dim; ++d) {
            float xv = __bfloat162float(x[off + d]);
            float wv = __bfloat162float(w[d]);
            float go = grad_out[off + d];
            float dx = rrms * (go * wv) - xv * rrc_d * gdxw;
            grad_x[off + d] += dx;
            atomicAdd(&grad_w[d], go * xv * rrms);
        }
    }
}



extern "C" __global__ void repeat_kv_bf16(
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ output,
    int batch, int num_kv_heads, int repeats, int seq_len, int head_dim
) {
    int num_q_heads = num_kv_heads * repeats;
    int total_elements = batch * num_q_heads * seq_len * head_dim;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < total_elements) {
        int d = idx % head_dim;
        int s = (idx / head_dim) % seq_len;
        int q_h = (idx / (head_dim * seq_len)) % num_q_heads;
        int b = idx / (head_dim * seq_len * num_q_heads);

        int kv_h = q_h / repeats;
        int in_idx = ((b * num_kv_heads + kv_h) * seq_len + s) * head_dim + d;

        output[idx] = input[in_idx];
    }
}

extern "C" __global__ void repeat_kv_backward_bf16(
    const __nv_bfloat16* __restrict__ grad_out,
    __nv_bfloat16* __restrict__ grad_in,
    int batch, int num_kv_heads, int repeats, int seq_len, int head_dim
) {
    int total_kv_elements = batch * num_kv_heads * seq_len * head_dim;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < total_kv_elements) {
        int d = idx % head_dim;
        int s = (idx / head_dim) % seq_len;
        int kv_h = (idx / (head_dim * seq_len)) % num_kv_heads;
        int b = idx / (head_dim * seq_len * num_kv_heads);

        // ALWAYS accumulate gradients in FP32 to avoid vanishing gradients
        float sum = 0.0f;
        int num_q_heads = num_kv_heads * repeats;
        
        for (int r = 0; r < repeats; ++r) {
            int q_h = kv_h * repeats + r;
            int gout_idx = ((b * num_q_heads + q_h) * seq_len + s) * head_dim + d;
            sum += __bfloat162float(grad_out[gout_idx]);
        }

        // Add back to existing gradient and cast down to BF16
        float current_grad = __bfloat162float(grad_in[idx]);
        grad_in[idx] = __float2bfloat16(current_grad + sum);
    }
}

extern "C" __global__ void cross_entropy_bf16(
    const __nv_bfloat16* logits, const float* targets, float* out_loss,
    const size_t batch_size, const size_t num_classes
) {
    size_t b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b < batch_size) {
        float max_val = -1e20f;
        for (size_t c = 0; c < num_classes; ++c) {
            // Explicitly unpack bf16 to f32 register
            float val = __bfloat162float(logits[b * num_classes + c]);
            if (val > max_val) max_val = val;
        }
        
        float sum_exp = 0.0f;
        for (size_t c = 0; c < num_classes; ++c) {
            float val = __bfloat162float(logits[b * num_classes + c]);
            sum_exp += expf(val - max_val);
        }
        
        size_t target_class = (size_t)targets[b];
        float target_val = __bfloat162float(logits[b * num_classes + target_class]);
        float prob = expf(target_val - max_val) / sum_exp;
        
        // out_loss is a standard float*, so f32 atomicAdd is perfectly fine
        atomicAdd(out_loss, -logf(prob + 1e-8f) / (float)batch_size);
    }
}

extern "C" __global__ void cross_entropy_backward_bf16(
    const __nv_bfloat16* logits, const float* targets, const float* grad_out,
    __nv_bfloat16* grad_logits, const size_t batch_size, const size_t num_classes
) {
    size_t b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b < batch_size) {
        float max_val = -1e20f;
        for (size_t c = 0; c < num_classes; ++c) {
            float val = __bfloat162float(logits[b * num_classes + c]);
            if (val > max_val) max_val = val;
        }
        
        float sum_exp = 0.0f;
        for (size_t c = 0; c < num_classes; ++c) {
            float val = __bfloat162float(logits[b * num_classes + c]);
            sum_exp += expf(val - max_val);
        }
        
        size_t target_class = (size_t)targets[b];
        float go = grad_out[0] / (float)batch_size; // grad_out is scalar f32
        
        for (size_t c = 0; c < num_classes; ++c) {
            float val = __bfloat162float(logits[b * num_classes + c]);
            float prob = expf(val - max_val) / sum_exp;
            float g = prob;
            if (c == target_class) g -= 1.0f;
            
            // Thread-safe read-add-write (No BF16 atomics needed)
            size_t idx = b * num_classes + c;
            float current_grad = __bfloat162float(grad_logits[idx]);
            grad_logits[idx] = __float2bfloat16(current_grad + (g * go));
        }
    }
}

// ── MATMUL BF16 OUTPUT ───────────────────────────────────────────────────────
extern "C" __global__ void matmul_bf16(
    const __nv_bfloat16* a, const __nv_bfloat16* b, float* out,
    const size_t m, const size_t k, const size_t n
) {
    __shared__ __nv_bfloat16 tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ __nv_bfloat16 tile_B[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;

    for (int t = 0; t < ((int)k + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        int a_col = t * TILE_SIZE + threadIdx.x;
        int b_row = t * TILE_SIZE + threadIdx.y;

        tile_A[threadIdx.y][threadIdx.x] = (row < (int)m && a_col < (int)k)
            ? a[row * k + a_col]
            : __float2bfloat16(0.0f);
        tile_B[threadIdx.y][threadIdx.x] = (b_row < (int)k && col < (int)n)
            ? b[b_row * n + col]
            : __float2bfloat16(0.0f);

        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE_SIZE; ++i) {
            sum += __bfloat162float(tile_A[threadIdx.y][i]) * __bfloat162float(tile_B[i][threadIdx.x]);
        }

        __syncthreads();
    }

    if (row < (int)m && col < (int)n) {
        out[row * n + col] = sum;
    }
}

// Mixed Precision MatMul: BF16 Inputs -> FP32 Accumulator/Output
extern "C" __global__ void matmul_bf16_f32(
    const __nv_bfloat16* a, const __nv_bfloat16* b, float* out,
    const size_t m, const size_t k, const size_t n
) {
    __shared__ __nv_bfloat16 tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ __nv_bfloat16 tile_B[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;

    for (int t = 0; t < ((int)k + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        int a_col = t * TILE_SIZE + threadIdx.x;
        int b_row = t * TILE_SIZE + threadIdx.y;

        tile_A[threadIdx.y][threadIdx.x] = (row < (int)m && a_col < (int)k)
            ? a[row * k + a_col]
            : __float2bfloat16(0.0f);
        tile_B[threadIdx.y][threadIdx.x] = (b_row < (int)k && col < (int)n)
            ? b[b_row * n + col]
            : __float2bfloat16(0.0f);

        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE_SIZE; ++i) {
            sum += __bfloat162float(tile_A[threadIdx.y][i]) * __bfloat162float(tile_B[i][threadIdx.x]);
        }

        __syncthreads();
    }

    if (row < (int)m && col < (int)n) {
        out[row * n + col] = sum;
    }
}

// Mixed Precision MatMul without full BF16 staging buffers.
//
// Inputs remain FP32 in global memory. Each multiply converts both operands
// to BF16 and accumulates in FP32, matching the numerical intent of
// `matmul_bf16_f32` while avoiding whole-matrix BF16 temp allocations.
extern "C" __global__ void matmul_f32_bf16accum_f32(
    const float* a, const float* b, float* out,
    const size_t m, const size_t k, const size_t n
) {
    __shared__ __nv_bfloat16 tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ __nv_bfloat16 tile_B[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;

    for (int t = 0; t < ((int)k + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        int a_col = t * TILE_SIZE + threadIdx.x;
        int b_row = t * TILE_SIZE + threadIdx.y;

        tile_A[threadIdx.y][threadIdx.x] = (row < (int)m && a_col < (int)k)
            ? __float2bfloat16(a[row * k + a_col])
            : __float2bfloat16(0.0f);
        tile_B[threadIdx.y][threadIdx.x] = (b_row < (int)k && col < (int)n)
            ? __float2bfloat16(b[b_row * n + col])
            : __float2bfloat16(0.0f);

        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE_SIZE; ++i) {
            sum += __bfloat162float(tile_A[threadIdx.y][i]) * __bfloat162float(tile_B[i][threadIdx.x]);
        }

        __syncthreads();
    }

    if (row < (int)m && col < (int)n) {
        out[row * n + col] = sum;
    }
}

extern "C" __global__ void matmul_f32_bf16rhsaccum_f32(
    const float* a, const __nv_bfloat16* b, float* out,
    const size_t m, const size_t k, const size_t n
) {
    __shared__ __nv_bfloat16 tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ __nv_bfloat16 tile_B[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;

    for (int t = 0; t < ((int)k + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        int a_col = t * TILE_SIZE + threadIdx.x;
        int b_row = t * TILE_SIZE + threadIdx.y;

        tile_A[threadIdx.y][threadIdx.x] = (row < (int)m && a_col < (int)k)
            ? __float2bfloat16(a[row * k + a_col])
            : __float2bfloat16(0.0f);
        tile_B[threadIdx.y][threadIdx.x] = (b_row < (int)k && col < (int)n)
            ? b[b_row * n + col]
            : __float2bfloat16(0.0f);

        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE_SIZE; ++i) {
            sum += __bfloat162float(tile_A[threadIdx.y][i]) * __bfloat162float(tile_B[i][threadIdx.x]);
        }

        __syncthreads();
    }

    if (row < (int)m && col < (int)n) {
        out[row * n + col] = sum;
    }
}

extern "C" __global__ void matmul_backward_a_bf16b_f32(
    const float* grad_out, const __nv_bfloat16* b, float* grad_a,
    const size_t m, const size_t k, const size_t n
) {
    __shared__ float tileGO[TILE_SIZE][TILE_SIZE];
    __shared__ float tileBT[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;

    for (int t = 0; t < ((int)n + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        int go_col = t * TILE_SIZE + threadIdx.x;
        int bt_n   = t * TILE_SIZE + threadIdx.y;

        tileGO[threadIdx.y][threadIdx.x] = (row < (int)m && go_col < (int)n)
            ? grad_out[row * n + go_col] : 0.0f;

        tileBT[threadIdx.y][threadIdx.x] = (col < (int)k && bt_n < (int)n)
            ? __bfloat162float(b[col * n + bt_n]) : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE_SIZE; ++i)
            sum += tileGO[threadIdx.y][i] * tileBT[i][threadIdx.x];

        __syncthreads();
    }

    if (row < (int)m && col < (int)k)
        grad_a[row * k + col] += sum;
}

// Forward: C = A * B^T
extern "C" __global__ void matmul_trans_b_bf16(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ C,
    unsigned long long M,
    unsigned long long K,
    unsigned long long N
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N) {
        // MUST accumulate in FP32 to prevent vanishing gradients/overflow
        float sum = 0.0f; 
        for (int i = 0; i < K; ++i) {
            float a_val = __bfloat162float(A[row * K + i]);
            float b_val = __bfloat162float(B[col * K + i]);
            sum += a_val * b_val;
        }
        // Downcast back to BF16 for the write
        C[row * N + col] = __float2bfloat16(sum);
    }
}

// 2. Transposed A Matmul with Accumulation (dB = dC^T * A)
// Computes Out = A^T * B
// A is [K, M] in memory. B is [K, N] in memory. Out is [M, N].
extern "C" __global__ void matmul_trans_a_bf16(
    const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* out,
    const size_t m, const size_t k, const size_t n
) {
    __shared__ __nv_bfloat16 tile_A[TILE_SIZE][TILE_SIZE]; 
    __shared__ __nv_bfloat16 tile_B[TILE_SIZE][TILE_SIZE]; 

    int row = blockIdx.y * TILE_SIZE + threadIdx.y; // m dimension
    int col = blockIdx.x * TILE_SIZE + threadIdx.x; // n dimension
    float sum = 0.0f;

    for (int t = 0; t < ((int)k + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Coalesced load of A (K x M matrix)
        int a_row_load = t * TILE_SIZE + threadIdx.y;
        int a_col_load = blockIdx.y * TILE_SIZE + threadIdx.x;
        tile_A[threadIdx.y][threadIdx.x] = (a_row_load < (int)k && a_col_load < (int)m)
            ? a[a_row_load * m + a_col_load] : __float2bfloat16(0.0f);

        // Coalesced load of B (K x N matrix)
        int b_row_load = t * TILE_SIZE + threadIdx.y;
        int b_col_load = blockIdx.x * TILE_SIZE + threadIdx.x;
        tile_B[threadIdx.y][threadIdx.x] = (b_row_load < (int)k && b_col_load < (int)n)
            ? b[b_row_load * n + b_col_load] : __float2bfloat16(0.0f);

        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE_SIZE; ++i) {
            // Read A transposed from shared memory by swapping the thread index lookup
            sum += __bfloat162float(tile_A[i][threadIdx.y]) * __bfloat162float(tile_B[i][threadIdx.x]);
        }
        __syncthreads();
    }

    if (row < (int)m && col < (int)n) {
        float current_grad = __bfloat162float(out[row * n + col]);
        out[row * n + col] = __float2bfloat16(current_grad + sum);
    }
}
extern "C" __global__ void matmul_f32_bf16out(
    const float* a, const float* b, __nv_bfloat16* out,
    const size_t m, const size_t k, const size_t n
) {
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;

    for (int t = 0; t < ((int)k + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        int a_col = t * TILE_SIZE + threadIdx.x;
        int b_row = t * TILE_SIZE + threadIdx.y;

        tile_A[threadIdx.y][threadIdx.x] = (row < (int)m && a_col < (int)k)
            ? a[row * k + a_col] : 0.0f;
        tile_B[threadIdx.y][threadIdx.x] = (b_row < (int)k && col < (int)n)
            ? b[b_row * n + col] : 0.0f;
        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE_SIZE; ++i)
            sum += tile_A[threadIdx.y][i] * tile_B[i][threadIdx.x];
        __syncthreads();
    }
    if (row < (int)m && col < (int)n)
        out[row * n + col] = __float2bfloat16(sum);
}

// Backward A: dA = dC * B^T
// dC is [M, N], B is [K, N]. dA is [M, K]
extern "C" __global__ void matmul_backward_a_bf16(
    const __nv_bfloat16* grad_out, const __nv_bfloat16* b, __nv_bfloat16* grad_a,
    const size_t m, const size_t k, const size_t n
) {
    __shared__ __nv_bfloat16 tileGO[TILE_SIZE][TILE_SIZE];
    __shared__ __nv_bfloat16 tileBT[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;

    for (int t = 0; t < ((int)n + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        int go_col = t * TILE_SIZE + threadIdx.x;
        int bt_n   = t * TILE_SIZE + threadIdx.y;

        tileGO[threadIdx.y][threadIdx.x] = (row < (int)m && go_col < (int)n)
            ? grad_out[row * n + go_col] : __float2bfloat16(0.0f);

        tileBT[threadIdx.y][threadIdx.x] = (col < (int)k && bt_n < (int)n)
            ? b[col * n + bt_n] : __float2bfloat16(0.0f);

        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE_SIZE; ++i) {
            sum += __bfloat162float(tileGO[threadIdx.y][i]) * __bfloat162float(tileBT[i][threadIdx.x]);
        }
        __syncthreads();
    }

    if (row < (int)m && col < (int)k) {
        // Accumulate into the existing gradient
        float current_grad = __bfloat162float(grad_a[row * k + col]);
        grad_a[row * k + col] = __float2bfloat16(current_grad + sum);
    }
}

// Backward B: dB = A^T * dC
// A is [M, K], dC is [M, N]. dB is [K, N]
extern "C" __global__ void matmul_backward_b_bf16(
    const __nv_bfloat16* a, const __nv_bfloat16* grad_out, __nv_bfloat16* grad_b,
    const size_t m, const size_t k, const size_t n
) {
    __shared__ __nv_bfloat16 tileAT[TILE_SIZE][TILE_SIZE];
    __shared__ __nv_bfloat16 tileGO[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y; // K dimension
    int col = blockIdx.x * TILE_SIZE + threadIdx.x; // N dimension
    float sum = 0.0f;

    for (int t = 0; t < ((int)m + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        int at_m = t * TILE_SIZE + threadIdx.x;
        int go_m = t * TILE_SIZE + threadIdx.y;

        tileAT[threadIdx.y][threadIdx.x] = (row < (int)k && at_m < (int)m)
            ? a[at_m * k + row] : __float2bfloat16(0.0f);

        tileGO[threadIdx.y][threadIdx.x] = (go_m < (int)m && col < (int)n)
            ? grad_out[go_m * n + col] : __float2bfloat16(0.0f);

        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE_SIZE; ++i) {
            sum += __bfloat162float(tileAT[threadIdx.y][i]) * __bfloat162float(tileGO[i][threadIdx.x]);
        }
        __syncthreads();
    }

    if (row < (int)k && col < (int)n) {
        // Accumulate into the existing gradient
        float current_grad = __bfloat162float(grad_b[row * n + col]);
        grad_b[row * n + col] = __float2bfloat16(current_grad + sum);
    }
}



// BMM implementations
extern "C" __global__ void bmm_f32_bf16accum_f32(
    const float* a, const float* b, float* out,
    const size_t batch, const size_t m, const size_t k, const size_t n,
    const bool trans_b
) {
    __shared__ __nv_bfloat16 tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ __nv_bfloat16 tile_B[TILE_SIZE][TILE_SIZE];

    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int b_idx = blockIdx.z;
    if (b_idx >= (int)batch) return;

    const float* a_batch = a + b_idx * (m * k);
    const float* b_batch = b + b_idx * (trans_b ? (n * k) : (k * n));
    float sum = 0.0f;

    for (int t = 0; t < ((int)k + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        int a_col = t * TILE_SIZE + threadIdx.x;
        int k_idx = t * TILE_SIZE + threadIdx.y;

        tile_A[threadIdx.y][threadIdx.x] = (row < (int)m && a_col < (int)k)
            ? __float2bfloat16(a_batch[row * k + a_col])
            : __float2bfloat16(0.0f);

        float b_val = 0.0f;
        if (col < (int)n && k_idx < (int)k) {
            b_val = trans_b ? b_batch[col * k + k_idx] : b_batch[k_idx * n + col];
        }
        tile_B[threadIdx.y][threadIdx.x] = __float2bfloat16(b_val);

        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE_SIZE; ++i) {
            sum += __bfloat162float(tile_A[threadIdx.y][i]) * __bfloat162float(tile_B[i][threadIdx.x]);
        }

        __syncthreads();
    }

    if (row < (int)m && col < (int)n) {
        out[b_idx * (m * n) + row * n + col] = sum;
    }
}

// ── BMM BF16 OUTPUT ──────────────────────────────────────────────────────────
// Batched Matrix Multiplication: BF16 Inputs -> FP32 Accumulator -> BF16 Output
// Handles both standard (A * B) and transposed B (A * B^T) via the trans_b flag.
extern "C" __global__ void bmm_bf16(
    const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* out,
    const size_t batch, const size_t m, const size_t k, const size_t n,
    const bool trans_b
) {
    __shared__ __nv_bfloat16 tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ __nv_bfloat16 tile_B[TILE_SIZE][TILE_SIZE];

    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int b_idx = blockIdx.z;
    if (b_idx >= (int)batch) return;

    const __nv_bfloat16* a_batch = a + b_idx * (m * k);
    // If trans_b is true, B is physically [N, K]. Otherwise, [K, N].
    const __nv_bfloat16* b_batch = b + b_idx * (trans_b ? (n * k) : (k * n));
    float sum = 0.0f;

    for (int t = 0; t < ((int)k + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        int a_col = t * TILE_SIZE + threadIdx.x;
        int k_idx = t * TILE_SIZE + threadIdx.y;

        tile_A[threadIdx.y][threadIdx.x] = (row < (int)m && a_col < (int)k)
            ? a_batch[row * k + a_col]
            : __float2bfloat16(0.0f);

        __nv_bfloat16 b_val = __float2bfloat16(0.0f);
        if (col < (int)n && k_idx < (int)k) {
            b_val = trans_b ? b_batch[col * k + k_idx] : b_batch[k_idx * n + col];
        }
        tile_B[threadIdx.y][threadIdx.x] = b_val;

        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE_SIZE; ++i) {
            sum += __bfloat162float(tile_A[threadIdx.y][i]) * __bfloat162float(tile_B[i][threadIdx.x]);
        }

        __syncthreads();
    }

    if (row < (int)m && col < (int)n) {
        out[b_idx * (m * n) + row * n + col] = __float2bfloat16(sum);
    }
}
// The existing bmm_f32_bf16accum_f32 outputs FP32.
// We add a variant that writes BF16 output after FP32 accumulation.

extern "C" __global__ void bmm_f32_bf16out(
    const float* a, const float* b, __nv_bfloat16* out,
    const size_t batch, const size_t m, const size_t k, const size_t n,
    const bool trans_b
) {
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE];

    int col   = blockIdx.x * TILE_SIZE + threadIdx.x;
    int row   = blockIdx.y * TILE_SIZE + threadIdx.y;
    int b_idx = blockIdx.z;
    if (b_idx >= (int)batch) return;

    const float* a_batch = a + b_idx * (m * k);
    const float* b_batch = b + b_idx * (trans_b ? (n * k) : (k * n));
    float sum = 0.0f;

    for (int t = 0; t < ((int)k + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        int a_col = t * TILE_SIZE + threadIdx.x;
        int k_idx = t * TILE_SIZE + threadIdx.y;

        tile_A[threadIdx.y][threadIdx.x] = (row < (int)m && a_col < (int)k)
            ? a_batch[row * k + a_col] : 0.0f;
        tile_B[threadIdx.y][threadIdx.x] = (col < (int)n && k_idx < (int)k)
            ? (trans_b ? b_batch[col * k + k_idx] : b_batch[k_idx * n + col]) : 0.0f;
        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE_SIZE; ++i)
            sum += tile_A[threadIdx.y][i] * tile_B[i][threadIdx.x];
        __syncthreads();
    }
    if (row < (int)m && col < (int)n)
        out[b_idx * (m * n) + row * n + col] = __float2bfloat16(sum);
}

// Backward A: dA = dC * B^T
extern "C" __global__ void bmm_backward_a_bf16(
    const __nv_bfloat16* grad_out, const __nv_bfloat16* b, __nv_bfloat16* grad_a,
    const size_t batch, const size_t m, const size_t k, const size_t n
) {
    __shared__ __nv_bfloat16 tileGO[TILE_SIZE][TILE_SIZE];
    __shared__ __nv_bfloat16 tileBT[TILE_SIZE][TILE_SIZE];

    int col = blockIdx.x * TILE_SIZE + threadIdx.x; // K dimension
    int row = blockIdx.y * TILE_SIZE + threadIdx.y; // M dimension
    int b_idx = blockIdx.z;

    if (b_idx >= (int)batch) return;

    const __nv_bfloat16* go_batch = grad_out + b_idx * (m * n);
    const __nv_bfloat16* b_batch  = b + b_idx * (k * n);
    __nv_bfloat16* ga_batch = grad_a + b_idx * (m * k);

    float sum = 0.0f;

    for (int t = 0; t < ((int)n + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        int go_col = t * TILE_SIZE + threadIdx.x;
        int bt_n   = t * TILE_SIZE + threadIdx.y;

        tileGO[threadIdx.y][threadIdx.x] = (row < (int)m && go_col < (int)n)
            ? go_batch[row * n + go_col] : __float2bfloat16(0.0f);

        // B is [K, N], but we are reading it as B^T which is [N, K].
        // threadIdx.y corresponds to the N dim, threadIdx.x to the K dim.
        tileBT[threadIdx.y][threadIdx.x] = (col < (int)k && bt_n < (int)n)
            ? b_batch[col * n + bt_n] : __float2bfloat16(0.0f);

        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE_SIZE; ++i) {
            sum += __bfloat162float(tileGO[threadIdx.y][i]) * __bfloat162float(tileBT[i][threadIdx.x]);
        }
        __syncthreads();
    }

    if (row < (int)m && col < (int)k) {
        float current_grad = __bfloat162float(ga_batch[row * k + col]);
        ga_batch[row * k + col] = __float2bfloat16(current_grad + sum);
    }
}

// Backward B: dB = A^T * dC
extern "C" __global__ void bmm_backward_b_bf16(
    const __nv_bfloat16* a, const __nv_bfloat16* grad_out, __nv_bfloat16* grad_b,
    const size_t batch, const size_t m, const size_t k, const size_t n
) {
    __shared__ __nv_bfloat16 tileAT[TILE_SIZE][TILE_SIZE];
    __shared__ __nv_bfloat16 tileGO[TILE_SIZE][TILE_SIZE];

    int col = blockIdx.x * TILE_SIZE + threadIdx.x; // N dimension
    int row = blockIdx.y * TILE_SIZE + threadIdx.y; // K dimension
    int b_idx = blockIdx.z;

    if (b_idx >= (int)batch) return;

    const __nv_bfloat16* a_batch  = a + b_idx * (m * k);
    const __nv_bfloat16* go_batch = grad_out + b_idx * (m * n);
    __nv_bfloat16* gb_batch = grad_b + b_idx * (k * n);

    float sum = 0.0f;

    for (int t = 0; t < ((int)m + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        int at_m = t * TILE_SIZE + threadIdx.x; 
        int go_m = t * TILE_SIZE + threadIdx.y; 

        // A is [M, K], reading as A^T which is [K, M].
        // row is K, at_m is M. Index in A memory is (at_m, row).
        tileAT[threadIdx.y][threadIdx.x] = (row < (int)k && at_m < (int)m)
            ? a_batch[at_m * k + row] : __float2bfloat16(0.0f);

        tileGO[threadIdx.y][threadIdx.x] = (go_m < (int)m && col < (int)n)
            ? go_batch[go_m * n + col] : __float2bfloat16(0.0f);

        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE_SIZE; ++i) {
            sum += __bfloat162float(tileAT[threadIdx.y][i]) * __bfloat162float(tileGO[i][threadIdx.x]);
        }
        __syncthreads();
    }

    if (row < (int)k && col < (int)n) {
        float current_grad = __bfloat162float(gb_batch[row * n + col]);
        gb_batch[row * n + col] = __float2bfloat16(current_grad + sum);
    }
}
// Backward A (Transposed B): dA = dC * B
extern "C" __global__ void bmm_backward_a_transb_bf16(
    const __nv_bfloat16* grad_out, const __nv_bfloat16* b, __nv_bfloat16* grad_a,
    const size_t batch, const size_t m, const size_t k, const size_t n
) {
    __shared__ __nv_bfloat16 tileGO[TILE_SIZE][TILE_SIZE];
    __shared__ __nv_bfloat16 tileB[TILE_SIZE][TILE_SIZE];

    int col = blockIdx.x * TILE_SIZE + threadIdx.x; // K dimension
    int row = blockIdx.y * TILE_SIZE + threadIdx.y; // M dimension
    int b_idx = blockIdx.z;

    if (b_idx >= (int)batch) return;

    const __nv_bfloat16* go_batch = grad_out + b_idx * (m * n);
    const __nv_bfloat16* b_batch  = b + b_idx * (n * k); // B is physically [N, K]
    __nv_bfloat16* ga_batch = grad_a + b_idx * (m * k);

    float sum = 0.0f;

    for (int t = 0; t < ((int)n + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        int go_col = t * TILE_SIZE + threadIdx.x; 
        int b_row  = t * TILE_SIZE + threadIdx.y; 

        tileGO[threadIdx.y][threadIdx.x] = (row < (int)m && go_col < (int)n)
            ? go_batch[row * n + go_col] : __float2bfloat16(0.0f);

        tileB[threadIdx.y][threadIdx.x] = (b_row < (int)n && col < (int)k)
            ? b_batch[b_row * k + col] : __float2bfloat16(0.0f);

        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE_SIZE; ++i) {
            sum += __bfloat162float(tileGO[threadIdx.y][i]) * __bfloat162float(tileB[i][threadIdx.x]);
        }
        __syncthreads();
    }

    if (row < (int)m && col < (int)k) {
        float current_grad = __bfloat162float(ga_batch[row * k + col]);
        ga_batch[row * k + col] = __float2bfloat16(current_grad + sum);
    }
}

// Backward B (Transposed B): dB = dC^T * A
extern "C" __global__ void bmm_backward_b_transb_bf16(
    const __nv_bfloat16* a, const __nv_bfloat16* grad_out, __nv_bfloat16* grad_b,
    const size_t batch, const size_t m, const size_t k, const size_t n
) {
    __shared__ __nv_bfloat16 tileGOT[TILE_SIZE][TILE_SIZE];
    __shared__ __nv_bfloat16 tileA[TILE_SIZE][TILE_SIZE];

    int col = blockIdx.x * TILE_SIZE + threadIdx.x; // K dimension
    int row = blockIdx.y * TILE_SIZE + threadIdx.y; // N dimension
    int b_idx = blockIdx.z;

    if (b_idx >= (int)batch) return;

    const __nv_bfloat16* a_batch  = a + b_idx * (m * k);
    const __nv_bfloat16* go_batch = grad_out + b_idx * (m * n);
    __nv_bfloat16* gb_batch = grad_b + b_idx * (n * k);

    float sum = 0.0f;

    for (int t = 0; t < ((int)m + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        int got_m = t * TILE_SIZE + threadIdx.x; 
        int a_row = t * TILE_SIZE + threadIdx.y; 

        // dC is physically [M, N]. We need dC^T which is [N, M].
        // row is N, got_m is M. Index in dC memory is (got_m, row).
        tileGOT[threadIdx.y][threadIdx.x] = (row < (int)n && got_m < (int)m)
            ? go_batch[got_m * n + row] : __float2bfloat16(0.0f);

        tileA[threadIdx.y][threadIdx.x] = (a_row < (int)m && col < (int)k)
            ? a_batch[a_row * k + col] : __float2bfloat16(0.0f);

        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE_SIZE; ++i) {
            sum += __bfloat162float(tileGOT[threadIdx.y][i]) * __bfloat162float(tileA[i][threadIdx.x]);
        }
        __syncthreads();
    }

    if (row < (int)n && col < (int)k) {
        float current_grad = __bfloat162float(gb_batch[row * k + col]);
        gb_batch[row * k + col] = __float2bfloat16(current_grad + sum);
    }
}

// ── GATHER (EMBEDDING) ───────────────────────────────────────────────────────

// BF16 weight table -> BF16 output (embedding in BF16 mode)
extern "C" __global__ void gather_bf16_bf16out(
    const __nv_bfloat16* weights,
    const float* indices,
    __nv_bfloat16* out,
    const size_t hidden_dim,
    const size_t out_size
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < out_size) {
        size_t row = i / hidden_dim;
        size_t col = i % hidden_dim;
        size_t w_row = (size_t)indices[row];
        out[i] = weights[w_row * hidden_dim + col];
    }
}

// gather backward is FP32 accumulation into param grad — unchanged (gather_backward_f32).

extern "C" __global__ void gather_backward_bf16(
    const __nv_bfloat16* indices, 
    const __nv_bfloat16* grad_out, 
    __nv_bfloat16* grad_w,
    const size_t hidden_dim, 
    const size_t out_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= out_size) return;

    int row = (int)__bfloat162float(indices[idx / hidden_dim]);
    int col = idx % hidden_dim;

    __nv_bfloat16 g = grad_out[idx];
    atomicAdd_bf16(&grad_w[row * hidden_dim + col], g);
}

// Scale a bf16 tensor by a float32 scalar
extern "C" __global__ void scale_bf16(__nv_bfloat16* a, const float scale, const size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = __bfloat162float(a[idx]);
        a[idx] = __float2bfloat16(val * scale);
    }
}
