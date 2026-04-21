#include "common.cuh"
#define FLASH_MAX_HEAD_DIM 256
#define FIM_IGNORE_INDEX 0xFFFFFFFFu

extern "C" __global__ void cross_entropy_masked_f32(
    const float* logits,
    const float* targets,       // float-cast usize; IGNORE_INDEX → 4294967295.0
    float* out_loss,
    const float normalizer,     // 1.0 / valid_count
    const size_t batch_size,
    const size_t num_classes
) {
    size_t b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch_size) return;

    unsigned int target_class = (unsigned int)targets[b];
    if (target_class == FIM_IGNORE_INDEX) return;

    float max_val = -1e20f;
    for (size_t c = 0; c < num_classes; ++c) {
        float val = bf16q(logits[b * num_classes + c]);
        if (val > max_val) max_val = val;
    }
    float sum_exp = 0.0f;
    for (size_t c = 0; c < num_classes; ++c)
        sum_exp += expf(bf16q(logits[b * num_classes + c]) - max_val);

    float prob = expf(bf16q(logits[b * num_classes + target_class]) - max_val) / sum_exp;
    atomicAdd(out_loss, -logf(prob + 1e-8f) * normalizer);
}

extern "C" __global__ void cross_entropy_masked_backward_f32(
    const float* logits,
    const float* targets,
    const float* grad_out,
    float* grad_logits,
    const float normalizer,
    const size_t batch_size,
    const size_t num_classes
) {
    size_t b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch_size) return;

    unsigned int target_class = (unsigned int)targets[b];
    if (target_class == FIM_IGNORE_INDEX) return;

    float max_val = -1e20f;
    for (size_t c = 0; c < num_classes; ++c) {
        float val = bf16q(logits[b * num_classes + c]);
        if (val > max_val) max_val = val;
    }
    float sum_exp = 0.0f;
    for (size_t c = 0; c < num_classes; ++c)
        sum_exp += expf(bf16q(logits[b * num_classes + c]) - max_val);

    float go = bf16q(grad_out[0]) * normalizer;
    for (size_t c = 0; c < num_classes; ++c) {
        float prob = expf(bf16q(logits[b * num_classes + c]) - max_val) / sum_exp;
        float g = prob;
        if (c == (size_t)target_class) g -= 1.0f;
        atomicAdd(&grad_logits[b * num_classes + c], bf16q(g * go));
    }
}

// --- BROADCAST-AWARE ADDITION (Using Safe Grid-Stride) ---
extern "C" __global__ void add_f32(
    const float* a, const float* b, float* out,
    const size_t n, const size_t rank,
    const size_t s0, const size_t s1, const size_t s2,
    const size_t a0, const size_t a1, const size_t a2,
    const size_t b0, const size_t b1, const size_t b2
) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        size_t temp = i;
        size_t idx_a = 0;
        size_t idx_b = 0;
        if (rank > 2) { size_t c = temp % s2; temp /= s2; idx_a += c * a2; idx_b += c * b2; }
        if (rank > 1) { size_t c = temp % s1; temp /= s1; idx_a += c * a1; idx_b += c * b1; }
        if (rank > 0) { size_t c = temp % s0; temp /= s0; idx_a += c * a0; idx_b += c * b0; }
        out[i] = bf16q(bf16q(a[idx_a]) + bf16q(b[idx_b]));
    }
}

extern "C" __global__ void accumulate_f32(
    float* grad_target, const float* grad_out,
    const size_t n, const size_t rank,
    const size_t s0, const size_t s1, const size_t s2,
    const size_t t0, const size_t t1, const size_t t2
) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        size_t temp = i;
        size_t idx_t = 0;
        if (rank > 2) { size_t c = temp % s2; temp /= s2; idx_t += c * t2; }
        if (rank > 1) { size_t c = temp % s1; temp /= s1; idx_t += c * t1; }
        if (rank > 0) { size_t c = temp % s0; temp /= s0; idx_t += c * t0; }
        atomicAdd(&grad_target[idx_t], bf16q(grad_out[i]));
    }
}

// --- ELEMENT-WISE OPS (Safe Grid-Stride replacing dangerous float4) ---
extern "C" __global__ void fill_f32(float* data, const float value, const size_t n) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        data[i] = value;
    }
}

extern "C" __global__ void scale_f32(float* data, const float scale, const size_t n) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        data[i] = bf16q(bf16q(data[i]) * bf16q(scale));
    }
}

extern "C" __global__ void mul_f32(const float* a, const float* b, float* out, const size_t n) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        out[i] = bf16q(bf16q(a[i]) * bf16q(b[i]));
    }
}

extern "C" __global__ void mul_backward_f32(
    const float* a, const float* b, const float* grad_out,
    float* grad_a, float* grad_b, const size_t n
) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        grad_a[i] += bf16q(bf16q(b[i]) * bf16q(grad_out[i]));
        grad_b[i] += bf16q(bf16q(a[i]) * bf16q(grad_out[i]));
    }
}

extern "C" __global__ void copy_f32(const float* src, float* dst, const size_t n) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        dst[i] = bf16q(src[i]);
    }
}

extern "C" __global__ void silu_f32(const float* x, float* out, const size_t n) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        float val = bf16q(x[i]);
        out[i] = bf16q(val / (1.0f + expf(-val)));
    }
}

extern "C" __global__ void silu_backward_f32(
    const float* x, const float* grad_out, float* grad_x, const size_t n
) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        float val = bf16q(x[i]);
        float sig = bf16q(1.0f / (1.0f + expf(-val)));
        float silu = bf16q(val * sig);
        float d_silu = bf16q(silu + sig * (1.0f - silu));
        grad_x[i] += bf16q(bf16q(grad_out[i]) * d_silu);
    }
}

// ============================================================
// MATMUL REVERTED TO SAFE 1D TILING
// Restored to perfectly match the 16x16 block launch logic in Rust
// ============================================================
extern "C" __global__ void matmul_f32(
    const float* a, const float* b, float* out,
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
            sum += bf16q(tile_A[threadIdx.y][i]) * bf16q(tile_B[i][threadIdx.x]);

        __syncthreads();
    }

    if (row < (int)m && col < (int)n)
        out[row * n + col] = bf16q(sum);
}

// Backwards matmul passes restored identically for launch config stability
extern "C" __global__ void matmul_backward_a_f32(
    const float* grad_out, const float* b, float* grad_a,
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
            ? b[col * n + bt_n] : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE_SIZE; ++i)
            sum += bf16q(tileGO[threadIdx.y][i]) * bf16q(tileBT[i][threadIdx.x]);

        __syncthreads();
    }

    if (row < (int)m && col < (int)k)
        grad_a[row * k + col] += bf16q(sum);
}

extern "C" __global__ void matmul_backward_b_f32(
    const float* a, const float* grad_out, float* grad_b,
    const size_t m, const size_t k, const size_t n
) {
    __shared__ float tileAT[TILE_SIZE][TILE_SIZE];
    __shared__ float tileGO[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;

    for (int t = 0; t < ((int)m + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        int at_m = t * TILE_SIZE + threadIdx.x;
        int go_m = t * TILE_SIZE + threadIdx.y;

        tileAT[threadIdx.y][threadIdx.x] = (row < (int)k && at_m < (int)m)
            ? a[at_m * k + row] : 0.0f;
        tileGO[threadIdx.y][threadIdx.x] = (go_m < (int)m && col < (int)n)
            ? grad_out[go_m * n + col] : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE_SIZE; ++i)
            sum += bf16q(tileAT[threadIdx.y][i]) * bf16q(tileGO[i][threadIdx.x]);

        __syncthreads();
    }

    if (row < (int)k && col < (int)n)
        grad_b[row * n + col] += bf16q(sum);
}

// -------------------------------------------------------------
// SUM AND MAX REDUCTIONS (Restored to 1 Thread = 1 Output Mapping)
// -------------------------------------------------------------
extern "C" __global__ void sum_f32(
    const float* a, float* out,
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
        for (size_t k = 0; k < reduced_dim_size; ++k) sum += bf16q(a[base_idx + k * reduced_dim_stride]);
        out[i] = bf16q(sum);
    }
}

extern "C" __global__ void sum_backward_f32(
    const float* grad_out, float* grad_a,
    const size_t out_size, const size_t reduced_dim_size, const size_t reduced_dim_stride,
    const size_t out_rank, const size_t os0, const size_t os1, const size_t os2,
    const size_t is0, const size_t is1, const size_t is2
) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < out_size; i += blockDim.x * gridDim.x) {
        size_t temp = i; size_t base_idx = 0;
        if (out_rank > 2) { size_t c = temp % os2; temp /= os2; base_idx += c * is2; }
        if (out_rank > 1) { size_t c = temp % os1; temp /= os1; base_idx += c * is1; }
        if (out_rank > 0) { size_t c = temp % os0; temp /= os0; base_idx += c * is0; }
        float go = bf16q(grad_out[i]);
        for (size_t k = 0; k < reduced_dim_size; ++k)
            atomicAdd(&grad_a[base_idx + k * reduced_dim_stride], bf16q(go));
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
        size_t temp = i; size_t base_idx = 0;
        if (out_rank > 2) { size_t c = temp % os2; temp /= os2; base_idx += c * is2; }
        if (out_rank > 1) { size_t c = temp % os1; temp /= os1; base_idx += c * is1; }
        if (out_rank > 0) { size_t c = temp % os0; temp /= os0; base_idx += c * is0; }
        float max_val = -1e20f;
        for (size_t k = 0; k < reduced_dim_size; ++k) {
            float val = bf16q(a[base_idx + k * reduced_dim_stride]);
            if (val > max_val) max_val = val;
        }
        out[i] = bf16q(max_val);
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
        size_t temp = i; size_t base_idx = 0;
        if (out_rank > 2) { size_t c = temp % os2; temp /= os2; base_idx += c * is2; }
        if (out_rank > 1) { size_t c = temp % os1; temp /= os1; base_idx += c * is1; }
        if (out_rank > 0) { size_t c = temp % os0; temp /= os0; base_idx += c * is0; }
        float max_val = -1e20f; size_t best_k = 0;
        for (size_t k = 0; k < reduced_dim_size; ++k) {
            float val = bf16q(a[base_idx + k * reduced_dim_stride]);
            if (val > max_val) { max_val = val; best_k = k; }
        }
        atomicAdd(&grad_a[base_idx + best_k * reduced_dim_stride], bf16q(grad_out[i]));
    }
}

// -------------------------------------------------------------
// BMM, Gather, RMSNorm, Rope, Flash Attention untouched below
// to ensure complete stability while retaining standard logic
// -------------------------------------------------------------
extern "C" __global__ void bmm_f32(
    const float* a, const float* b, float* out,
    const size_t batch, const size_t m, const size_t k, const size_t n,
    const bool trans_b
) {
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE];

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
            ? bf16q(a_batch[row * k + a_col]) : 0.0f;
        tile_B[threadIdx.y][threadIdx.x] = (col < (int)n && k_idx < (int)k)
            ? bf16q(trans_b ? b_batch[col * k + k_idx] : b_batch[k_idx * n + col]) : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE_SIZE; ++i) {
            sum += bf16q(tile_A[threadIdx.y][i]) * bf16q(tile_B[i][threadIdx.x]);
        }
        __syncthreads();
    }
    if (row < (int)m && col < (int)n) {
        out[b_idx * (m * n) + row * n + col] = bf16q(sum);
    }
}

extern "C" __global__ void bmm_backward_a_f32(
    const float* __restrict__ grad_out, const float* __restrict__ b, float* __restrict__ grad_a,
    const size_t batch, const size_t m, const size_t k, const size_t n
) {
    __shared__ float s_go[TILE_SIZE][TILE_SIZE + 1];
    __shared__ float s_b [TILE_SIZE][TILE_SIZE + 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const size_t b_idx = blockIdx.z;
    const size_t row = blockIdx.y * TILE_SIZE + ty;
    const size_t col = blockIdx.x * TILE_SIZE + tx;

    const float* go_batch = grad_out + b_idx * (m * n);
    const float* b_batch  = b        + b_idx * (k * n);
    float* ga_batch       = grad_a   + b_idx * (m * k);

    float acc = 0.0f;
    for (size_t t0 = 0; t0 < n; t0 += TILE_SIZE) {
        if (row < m && (t0 + tx) < n) s_go[ty][tx] = bf16q(go_batch[row * n + (t0 + tx)]);
        else s_go[ty][tx] = 0.0f;

        if (col < k && (t0 + ty) < n) s_b[ty][tx] = bf16q(b_batch[col * n + (t0 + ty)]);
        else s_b[ty][tx] = 0.0f;
        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE_SIZE; ++i) acc += bf16q(s_go[ty][i]) * bf16q(s_b[i][tx]);
        __syncthreads();
    }
    if (row < m && col < k) ga_batch[row * k + col] = bf16q(acc);
}

extern "C" __global__ void bmm_backward_b_f32(
    const float* __restrict__ a, const float* __restrict__ grad_out, float* __restrict__ grad_b,
    const size_t batch, const size_t m, const size_t k, const size_t n
) {
    __shared__ float s_a [TILE_SIZE][TILE_SIZE + 1];
    __shared__ float s_go[TILE_SIZE][TILE_SIZE + 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const size_t b_idx = blockIdx.z;
    const size_t row = blockIdx.y * TILE_SIZE + ty; 
    const size_t col = blockIdx.x * TILE_SIZE + tx; 

    const float* a_batch  = a        + b_idx * (m * k);
    const float* go_batch = grad_out + b_idx * (m * n);
    float* gb_batch       = grad_b   + b_idx * (k * n);

    float acc = 0.0f;
    for (size_t t0 = 0; t0 < m; t0 += TILE_SIZE) {
        if (row < k && (t0 + tx) < m) s_a[ty][tx] = bf16q(a_batch[(t0 + tx) * k + row]);
        else s_a[ty][tx] = 0.0f;

        if (col < n && (t0 + ty) < m) s_go[ty][tx] = bf16q(go_batch[(t0 + ty) * n + col]);
        else s_go[ty][tx] = 0.0f;
        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE_SIZE; ++i) acc += bf16q(s_a[ty][i]) * bf16q(s_go[i][tx]);
        __syncthreads();
    }
    if (row < k && col < n) gb_batch[row * n + col] = bf16q(acc);
}

extern "C" __global__ void bmm_backward_a_transb_f32(
    const float* __restrict__ grad_out, const float* __restrict__ b, float* __restrict__ grad_a,
    const size_t batch, const size_t m, const size_t k, const size_t n
) {
    __shared__ float s_go[TILE_SIZE][TILE_SIZE + 1];
    __shared__ float s_b [TILE_SIZE][TILE_SIZE + 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const size_t b_idx = blockIdx.z;
    const size_t row = blockIdx.y * TILE_SIZE + ty;
    const size_t col = blockIdx.x * TILE_SIZE + tx;

    const float* go_batch = grad_out + b_idx * (m * n);
    const float* b_batch  = b        + b_idx * (n * k);
    float* ga_batch       = grad_a   + b_idx * (m * k);

    float acc = 0.0f;
    for (size_t t0 = 0; t0 < n; t0 += TILE_SIZE) {
        if (row < m && (t0 + tx) < n) s_go[ty][tx] = bf16q(go_batch[row * n + (t0 + tx)]);
        else s_go[ty][tx] = 0.0f;

        if (col < k && (t0 + ty) < n) s_b[ty][tx] = bf16q(b_batch[(t0 + ty) * k + col]);
        else s_b[ty][tx] = 0.0f;
        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE_SIZE; ++i) acc += bf16q(s_go[ty][i]) * bf16q(s_b[i][tx]);
        __syncthreads();
    }
    if (row < m && col < k) ga_batch[row * k + col] = bf16q(acc);
}

extern "C" __global__ void bmm_backward_b_transb_f32(
    const float* __restrict__ a, const float* __restrict__ grad_out, float* __restrict__ grad_b,
    const size_t batch, const size_t m, const size_t k, const size_t n
) {
    __shared__ float s_go[TILE_SIZE][TILE_SIZE + 1];
    __shared__ float s_a [TILE_SIZE][TILE_SIZE + 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const size_t b_idx = blockIdx.z;
    const size_t row = blockIdx.y * TILE_SIZE + ty;
    const size_t col = blockIdx.x * TILE_SIZE + tx;

    const float* a_batch  = a        + b_idx * (m * k);
    const float* go_batch = grad_out + b_idx * (m * n);
    float* gb_batch       = grad_b   + b_idx * (n * k);

    float acc = 0.0f;
    for (size_t t0 = 0; t0 < m; t0 += TILE_SIZE) {
        if (row < n && (t0 + tx) < m) s_go[ty][tx] = bf16q(go_batch[(t0 + tx) * n + row]);
        else s_go[ty][tx] = 0.0f;

        if (col < k && (t0 + ty) < m) s_a[ty][tx] = bf16q(a_batch[(t0 + ty) * k + col]);
        else s_a[ty][tx] = 0.0f;
        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE_SIZE; ++i) acc += bf16q(s_go[ty][i]) * bf16q(s_a[i][tx]);
        __syncthreads();
    }
    if (row < n && col < k) gb_batch[row * k + col] = bf16q(acc);
}

extern "C" __global__ void gather_f32(
    const float* weights, const float* indices, float* out,
    const size_t hidden_dim, const size_t out_size
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < out_size) {
        size_t row = i / hidden_dim;
        size_t col = i % hidden_dim;
        size_t w_row = (size_t)indices[row];
        out[i] = weights[w_row * hidden_dim + col];
    }
}

extern "C" __global__ void gather_backward_f32(
    const float* indices, const float* grad_out, float* grad_weights,
    const size_t hidden_dim, const size_t out_size
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < out_size) {
        size_t row = i / hidden_dim;
        size_t col = i % hidden_dim;
        size_t w_row = (size_t)indices[row];
        atomicAdd(&grad_weights[w_row * hidden_dim + col], grad_out[i]);
    }
}

extern "C" __global__ void rmsnorm_f32(
    const float* x, const float* w, float* out,
    const size_t dim, const float eps, const size_t num_vecs
) {
    size_t n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n < num_vecs) {
        size_t offset = n * dim;
        float sum_sq = 0.0f;
        for (size_t d = 0; d < dim; ++d) { float val = bf16q(x[offset + d]); sum_sq += bf16q(val * val); }
        float rrms = rsqrtf(sum_sq / (float)dim + eps);
        for (size_t d = 0; d < dim; ++d) { out[offset + d] = bf16q(bf16q(x[offset + d]) * bf16q(rrms) * bf16q(w[d])); }
    }
}

extern "C" __global__ void rmsnorm_backward_f32(
    const float* x, const float* w, const float* grad_out,
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
            grad_dot_x_w += bf16q(grad_out[offset + d]) * bf16q(x[offset + d]) * bf16q(w[d]);
        }
        float rrc_d = (rrms * rrms * rrms) / (float)dim;
        for (size_t d = 0; d < dim; ++d) {
            float val_x = bf16q(x[offset + d]);
            float val_w = bf16q(w[d]);
            float go = bf16q(grad_out[offset + d]);
            float dx = bf16q(rrms * bf16q(go * val_w) - bf16q(val_x * rrc_d * grad_dot_x_w));
            grad_x[offset + d] += dx;
            atomicAdd(&grad_w[d], bf16q(go * val_x * rrms));
        }
    }
}

extern "C" __global__ void cross_entropy_f32(
    const float* logits, const float* targets, float* out_loss,
    const size_t batch_size, const size_t num_classes
) {
    size_t b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b < batch_size) {
        float max_val = -1e20f;
        for (size_t c = 0; c < num_classes; ++c) {
            float val = bf16q(logits[b * num_classes + c]);
            if (val > max_val) max_val = val;
        }
        float sum_exp = 0.0f;
        for (size_t c = 0; c < num_classes; ++c)
            sum_exp += expf(bf16q(logits[b * num_classes + c]) - max_val);
        size_t target_class = (size_t)targets[b];
        float prob = expf(bf16q(logits[b * num_classes + target_class]) - max_val) / sum_exp;
        atomicAdd(out_loss, -logf(prob + 1e-8f) / (float)batch_size);
    }
}

extern "C" __global__ void cross_entropy_backward_f32(
    const float* logits, const float* targets, const float* grad_out,
    float* grad_logits, const size_t batch_size, const size_t num_classes
) {
    size_t b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b < batch_size) {
        float max_val = -1e20f;
        for (size_t c = 0; c < num_classes; ++c) {
            float val = bf16q(logits[b * num_classes + c]);
            if (val > max_val) max_val = val;
        }
        float sum_exp = 0.0f;
        for (size_t c = 0; c < num_classes; ++c)
            sum_exp += expf(bf16q(logits[b * num_classes + c]) - max_val);
        size_t target_class = (size_t)targets[b];
        float go = bf16q(grad_out[0]) / (float)batch_size;
        for (size_t c = 0; c < num_classes; ++c) {
            float prob = expf(bf16q(logits[b * num_classes + c]) - max_val) / sum_exp;
            float g = prob;
            if (c == target_class) g -= 1.0f;
            atomicAdd(&grad_logits[b * num_classes + c], bf16q(g * go));
        }
    }
}

extern "C" __global__ void softmax_f32(const float* x, float* out, const size_t B, const size_t N) {
    size_t b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b < B) {
        float max_val = -1e20f;
        for (size_t i = 0; i < N; ++i) if (bf16q(x[b * N + i]) > max_val) max_val = bf16q(x[b * N + i]);
        float sum = 0.0f;
        for (size_t i = 0; i < N; ++i) { float e = expf(bf16q(x[b * N + i]) - max_val); out[b * N + i] = bf16q(e); sum += e; }
        for (size_t i = 0; i < N; ++i) out[b * N + i] = bf16q(out[b * N + i] / sum);
    }
}

extern "C" __global__ void softmax_backward_f32(const float* out, const float* grad_out, float* grad_x, const size_t B, const size_t N) {
    size_t b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b < B) {
        float sum_go = 0.0f;
        for (size_t i = 0; i < N; ++i) sum_go += bf16q(out[b * N + i]) * bf16q(grad_out[b * N + i]);
        for (size_t i = 0; i < N; ++i) {
            float g = bf16q(bf16q(out[b * N + i]) * bf16q(bf16q(grad_out[b * N + i]) - sum_go));
            atomicAdd(&grad_x[b * N + i], g);
        }
    }
}

extern "C" __global__ void flash_attention_f32(
    const float* q, const float* k, const float* v, float* out,
    const size_t batch, const size_t m, const size_t n, const size_t d,
    const float scale, const bool causal
) {
    size_t row_idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_rows = batch * m;
    if (row_idx >= total_rows || d > FLASH_MAX_HEAD_DIM) return;

    size_t b = row_idx / m;
    size_t i = row_idx % m;

    const float* q_row = q + b * (m * d) + i * d;
    const float* k_batch = k + b * (n * d);
    const float* v_batch = v + b * (n * d);
    float* out_row = out + b * (m * d) + i * d;

    float acc[FLASH_MAX_HEAD_DIM];
    for (size_t x = 0; x < d; ++x) acc[x] = 0.0f;

    float running_max = -1e20f;
    float running_l = 0.0f;

    for (size_t j = 0; j < n; ++j) {
        float score = 0.0f;
        const float* k_row = k_batch + j * d;
        for (size_t x = 0; x < d; ++x) score += bf16q(q_row[x]) * bf16q(k_row[x]);
        score = bf16q(score * bf16q(scale));
        if (causal && j > i) score = -1e20f;

        float new_max = fmaxf(running_max, score);
        float alpha = expf(running_max - new_max);
        float p = expf(score - new_max);
        float new_l = running_l * alpha + p;

        const float* v_row = v_batch + j * d;
        for (size_t x = 0; x < d; ++x) acc[x] = bf16q(acc[x] * alpha + p * bf16q(v_row[x]));

        running_max = new_max;
        running_l = new_l;
    }

    float inv_l = 1.0f / fmaxf(running_l, 1e-9f);
    for (size_t x = 0; x < d; ++x) out_row[x] = bf16q(acc[x] * inv_l);
}

extern "C" __global__ void transpose_0213_f32(
    const float* src, float* dst,
    const size_t B, const size_t S, const size_t H, const size_t D
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = B * S * H * D;
    if (i < total) {
        size_t d = i % D; size_t temp = i / D;
        size_t h = temp % H; temp /= H;
        size_t s = temp % S; size_t b = temp / S;
        size_t dst_idx = b * (H * S * D) + h * (S * D) + s * D + d;
        dst[dst_idx] = bf16q(src[i]);
    }
}

extern "C" __global__ void transpose_0213_backward_f32(
    const float* grad_out, float* grad_src,
    const size_t B, const size_t S, const size_t H, const size_t D
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = B * S * H * D;
    if (i < total) {
        size_t d = i % D; size_t temp = i / D;
        size_t h = temp % H; temp /= H;
        size_t s = temp % S; size_t b = temp / S;
        size_t dst_idx = b * (H * S * D) + h * (S * D) + s * D + d;
        grad_src[i] += bf16q(grad_out[dst_idx]);
    }
}

extern "C" __global__ void rope_f32(
    const float* x, float* out,
    const size_t seq_len, const size_t hidden_dim, const size_t head_dim, const size_t num_pairs
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_pairs) {
        size_t d_pair = i % (hidden_dim / 2);
        size_t d_head_pair = d_pair % (head_dim / 2);
        size_t seq_idx = (i / (hidden_dim / 2)) % seq_len;
        size_t idx1 = i * 2; size_t idx2 = i * 2 + 1;
        float freq = powf(10000.0f, -2.0f * (float)d_head_pair / (float)head_dim);
        float angle = (float)seq_idx * freq;
        float cos_a = cosf(angle); float sin_a = sinf(angle);
        float x1 = bf16q(x[idx1]); float x2 = bf16q(x[idx2]);
        out[idx1] = bf16q(x1 * cos_a - x2 * sin_a);
        out[idx2] = bf16q(x2 * cos_a + x1 * sin_a);
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
        size_t idx1 = i * 2; size_t idx2 = i * 2 + 1;
        float freq = powf(10000.0f, -2.0f * (float)d_head_pair / (float)head_dim);
        float angle = (float)seq_idx * freq;
        float cos_a = cosf(angle); float sin_a = sinf(angle);
        float g1 = bf16q(grad_out[idx1]); float g2 = bf16q(grad_out[idx2]);
        grad_x[idx1] += bf16q(g1 * cos_a + g2 * sin_a);
        grad_x[idx2] += bf16q(g2 * cos_a - g1 * sin_a);
    }
}

extern "C" __global__ void adamw_step_f32(
    float* weights, const float* grads, float* m, float* v,
    const float lr, const float beta1, const float beta2,
    const float eps, const float weight_decay,
    const float bc1, const float bc2, const size_t n
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float g = grads[i];
        float w = weights[i];

        w *= (1.0f - lr * weight_decay);

        float m_new = beta1 * m[i] + (1.0f - beta1) * g;
        float v_new = beta2 * v[i] + (1.0f - beta2) * g * g;
        m[i] = m_new;
        v[i] = v_new;

        float m_hat = m_new * bc1;
        float v_hat = v_new * bc2;
        weights[i] = w - lr * m_hat / (sqrtf(v_hat) + eps);
    }
}


extern "C" __global__ void repeat_kv_f32(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch, int num_kv_heads, int repeats, int seq_len, int head_dim
) {
    int num_q_heads = num_kv_heads * repeats;
    int total_elements = batch * num_q_heads * seq_len * head_dim;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < total_elements) {
        // Map linear index to (B, Q_H, S, D)
        int d = idx % head_dim;
        int s = (idx / head_dim) % seq_len;
        int q_h = (idx / (head_dim * seq_len)) % num_q_heads;
        int b = idx / (head_dim * seq_len * num_q_heads);

        // Map Q_H back to KV_H
        int kv_h = q_h / repeats;
        int in_idx = ((b * num_kv_heads + kv_h) * seq_len + s) * head_dim + d;

        output[idx] = input[in_idx];
    }
}

extern "C" __global__ void repeat_kv_backward_f32(
    const float* __restrict__ grad_out,
    float* __restrict__ grad_in,
    int batch, int num_kv_heads, int repeats, int seq_len, int head_dim
) {
    int total_kv_elements = batch * num_kv_heads * seq_len * head_dim;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < total_kv_elements) {
        // Map linear index to (B, KV_H, S, D)
        int d = idx % head_dim;
        int s = (idx / head_dim) % seq_len;
        int kv_h = (idx / (head_dim * seq_len)) % num_kv_heads;
        int b = idx / (head_dim * seq_len * num_kv_heads);

        // Reduce gradients from the repeated Query Heads back into the KV Head
        float sum = 0.0f;
        int num_q_heads = num_kv_heads * repeats;
        for (int r = 0; r < repeats; ++r) {
            int q_h = kv_h * repeats + r;
            int gout_idx = ((b * num_q_heads + q_h) * seq_len + s) * head_dim + d;
            sum += grad_out[gout_idx];
        }
        
        // Accumulate (+=) into grad_in
        grad_in[idx] += sum; 
    }
}

// Standard tiled matrix multiplication for C = A * B^T
// A: [M, K], B: [N, K], C: [M, N]
extern "C" __global__ void matmul_trans_b_f32(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    unsigned long long M,
    unsigned long long K,
    unsigned long long N
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; ++i) {
            sum += A[row * K + i] * B[col * K + i];
        }
        C[row * N + col] = sum;
    }
}

// Backward kernel for B gradient: dB = dC^T * A
// dC: [M, N], A: [M, K], dB: [N, K]
extern "C" __global__ void matmul_trans_a_f32(
    const float* __restrict__ dC,
    const float* __restrict__ A,
    float* __restrict__ dB,
    unsigned long long M,
    unsigned long long N,
    unsigned long long K
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < N && col < K) {
        float sum = 0.0f;
        for (int i = 0; i < M; ++i) {
            sum += dC[i * N + row] * A[i * K + col];
        }
        dB[row * K + col] += sum;
    }
}
