#include "common.cuh"
#define FLASH_MAX_HEAD_DIM 256
#define FIM_IGNORE_INDEX 0xFFFFFFFFu

/**
 * Computes masked cross‑entropy loss for a batch of logits and targets.
 *
 * # Arguments
 * * `logits` – Logit matrix [batch_size, num_classes] in BF16 format.
 * * `targets` – Target indices cast to float; `IGNORE_INDEX` (4294967295.0f) marks padding.
 * * `out_loss` – Scalar output; loss accumulated via atomic addition.
 * * `normalizer` – 1 / valid_count used for averaging.
 * * `batch_size`, `num_classes` – Dimensions of the input tensors.
 *
 * # Notes
 * The kernel performs a numerically stable softmax and accumulates the negative log‑probability of the true class.
 */
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

/**
 * Backward pass for masked cross‑entropy loss.
 *
 * # Arguments
 * * `logits` – Logit matrix [batch_size, num_classes] in BF16 format.
 * * `targets` – Target indices cast to float; `IGNORE_INDEX` marks padding.
 * * `grad_out` – Gradient of the loss scalar.
 * * `grad_logits` – Gradient to accumulate into the logits matrix.
 * * `normalizer` – 1 / valid_count.
 * * `batch_size`, `num_classes` – Input dimensions.
 *
 * # Notes
 * Computes the gradient of the softmax cross‑entropy with respect to logits, ignoring padded positions.
 */
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
/**
 * Broadcast‑aware element‑wise addition of two float tensors.
 *
 * # Arguments
 * * `a`, `b` – Input tensors, possibly broadcasted.
 * * `out` – Output tensor.
 * * `n` – Total number of elements in the broadcasted result.
 * * `rank`, `s0`, `s1`, `s2` – Shape information of the broadcasted result.
 * * `a0`, `a1`, `a2` – Strides of tensor `a` in the broadcasted layout.
 * * `b0`, `b1`, `b2` – Strides of tensor `b` in the broadcasted layout.
 *
 * # Notes
 * Uses grid‑stride loop for arbitrary shapes up to rank 3; performs BF16 conversion for each operand.
 */
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

/**
 * Accumulates gradients into a target tensor respecting broadcasting.
 *
 * # Arguments
 * * `grad_target` – Gradient tensor to accumulate into.
 * * `grad_out` – Gradient output from the next operation.
 * * `n`, `rank`, `s0`, `s1`, `s2` – Size and shape of the broadcasted result.
 * * `t0`, `t1`, `t2` – Strides of the target tensor in the broadcasted layout.
 *
 * # Notes
 * Uses atomic adds and grid‑stride to handle arbitrary shapes up to rank 3.
 */
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
/**
 * Fills a tensor with a constant value.
 *
 * # Arguments
 * * `data` – Output tensor.
 * * `value` – Scalar value to fill.
 * * `n` – Number of elements.
 *
 * # Notes
 * Performs BF16 conversion on the value before storing.
 */
extern "C" __global__ void fill_f32(float* data, const float value, const size_t n) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        data[i] = value;
    }
}

/**
 * Scales each element of a tensor by a constant factor.
 *
 * # Arguments
 * * `data` – Tensor to scale in-place.
 * * `scale` – Scaling factor.
 * * `n` – Number of elements.
 *
 * # Notes
 * Both input and output are converted to BF16; atomic add is not used.
 */
extern "C" __global__ void scale_f32(float* data, const float scale, const size_t n) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        data[i] = bf16q(bf16q(data[i]) * bf16q(scale));
    }
}

/**
 * Element‑wise multiplication of two tensors.
 *
 * # Arguments
 * * `a`, `b` – Input tensors.
 * * `out` – Result tensor.
 * * `n` – Number of elements.
 *
 * # Notes
 * Uses grid‑stride and BF16 conversion for each operand.
 */
extern "C" __global__ void mul_f32(const float* a, const float* b, float* out, const size_t n) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        out[i] = bf16q(bf16q(a[i]) * bf16q(b[i]));
    }
}

/**
 * Backward pass for element‑wise multiplication.
 *
 * # Arguments
 * * `a`, `b` – Original inputs.
 * * `grad_out` – Gradient from the next operation.
 * * `grad_a`, `grad_b` – Gradients to accumulate into.
 * * `n` – Number of elements.
 *
 * # Notes
 * Accumulates gradients using atomic adds with BF16 conversion.
 */
extern "C" __global__ void mul_backward_f32(
    const float* a, const float* b, const float* grad_out,
    float* grad_a, float* grad_b, const size_t n
) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        grad_a[i] += bf16q(bf16q(b[i]) * bf16q(grad_out[i]));
        grad_b[i] += bf16q(bf16q(a[i]) * bf16q(grad_out[i]));
    }
}

/**
 * Copies a tensor from source to destination.
 *
 * # Arguments
 * * `src` – Source tensor.
 * * `dst` – Destination tensor.
 * * `n` – Number of elements.
 *
 * # Notes
 * Performs BF16 conversion on each element.
 */
extern "C" __global__ void copy_f32(const float* src, float* dst, const size_t n) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        dst[i] = bf16q(src[i]);
    }
}

/**
 * Computes the SiLU activation function (x * sigmoid(x)).
 *
 * # Arguments
 * * `x` – Input tensor.
 * * `out` – Output tensor.
 * * `n` – Number of elements.
 *
 * # Notes
 * Implements the formula with BF16 conversion for each operation.
 */
extern "C" __global__ void silu_f32(const float* x, float* out, const size_t n) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        float val = bf16q(x[i]);
        out[i] = bf16q(val / (1.0f + expf(-val)));
    }
}

/**
 * Backward pass for the SiLU activation.
 *
 * # Arguments
 * * `x` – Original input tensor.
 * * `grad_out` – Gradient from the next operation.
 * * `grad_x` – Gradient to accumulate into.
 * * `n` – Number of elements.
 *
 * # Notes
 * Uses the derivative of SiLU: sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x)).
 */
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
/**
 * Tiled matrix multiplication: C = A × B.
 *
 * # Arguments
 * * `a` – Left matrix [m, k].
 * * `b` – Right matrix [k, n].
 * * `out` – Result matrix [m, n].
 * * `m`, `k`, `n` – Dimensions.
 *
 * # Notes
 * Uses shared memory tiling with TILE_SIZE and BF16 conversion; one thread per output element.
 */
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
/**
 * Backward pass for matrix multiplication w.r.t. matrix A (∂C/∂A).
 *
 * # Arguments
 * * `grad_out` – Gradient of the output matrix [m, n].
 * * `b` – Right matrix [k, n].
 * * `grad_a` – Gradient to accumulate into [m, k].
 * * `m`, `k`, `n` – Dimensions.
 *
 * # Notes
 * Accumulates using atomic adds; shared memory tiling is used.
 */
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

/**
 * Backward pass for matrix multiplication w.r.t. matrix B (∂C/∂B).
 *
 * # Arguments
 * * `a` – Left matrix [m, k].
 * * `grad_out` – Gradient of the output matrix [m, n].
 * * `grad_b` – Gradient to accumulate into [k, n].
 * * `m`, `k`, `n` – Dimensions.
 *
 * # Notes
 * Accumulates using atomic adds; shared memory tiling is used.
 */
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
/**
 * Computes a sum reduction along a specified dimension.
 *
 * # Arguments
 * * `a` – Input tensor.
 * * `out` – Output tensor of reduced shape.
 * * `out_size` – Number of elements in the output.
 * * `reduced_dim_size` – Size of the dimension being reduced.
 * * `reduced_dim_stride` – Stride to step along the reduced dimension.
 * * `out_rank`, `os0`, `os1`, `os2` – Shape of the output tensor.
 * * `is0`, `is1`, `is2` – Strides of the input tensor.
 *
 * # Notes
 * Handles tensors up to rank 3 and performs BF16 conversion.
 */
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

/**
 * Backward pass for sum reduction.
 *
 * # Arguments
 * * `grad_out` – Gradient of the output tensor.
 * * `grad_a` – Gradient to accumulate into the input tensor.
 * * `out_size`, `reduced_dim_size`, `reduced_dim_stride` – Same as in forward.
 * * `out_rank`, `os0`, `os1`, `os2`, `is0`, `is1`, `is2` – Shape and stride metadata.
 *
 * # Notes
 * Uses atomic adds and grid‑stride to accumulate gradients.
 */
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

/**
 * Computes a max reduction along a specified dimension.
 *
 * # Arguments
 * * `a` – Input tensor.
 * * `out` – Output tensor of reduced shape.
 * * `out_size` – Number of elements in the output.
 * * `reduced_dim_size` – Size of the dimension being reduced.
 * * `reduced_dim_stride` – Stride to step along the reduced dimension.
 * * `out_rank`, `os0`, `os1`, `os2` – Shape of the output tensor.
 * * `is0`, `is1`, `is2` – Strides of the input tensor.
 *
 * # Notes
 * Handles tensors up to rank 3; BF16 conversion is applied.
 */
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

/**
 * Backward pass for max reduction.
 *
 * # Arguments
 * * `a` – Input tensor.
 * * `grad_out` – Gradient of the output tensor.
 * * `grad_a` – Gradient to accumulate into the input tensor.
 * * `out_size`, `reduced_dim_size`, `reduced_dim_stride` – As in forward.
 * * `out_rank`, `os0`, `os1`, `os2`, `is0`, `is1`, `is2` – Shape and stride metadata.
 *
 * # Notes
 * Accumulates gradient only into the element that achieved the maximum.
 */
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
/**
 * Batched matrix multiplication: C[b] = A[b] × B[b].
 *
 * # Arguments
 * * `a` – Batch of left matrices [batch, m, k].
 * * `b` – Batch of right matrices [batch, k, n] (or transposed if `trans_b`).
 * * `out` – Batch of output matrices [batch, m, n].
 * * `batch`, `m`, `k`, `n` – Batch size and dimensions.
 * * `trans_b` – Whether `b` is stored transposed.
 *
 * # Notes
 * Uses shared memory tiling; one thread per output element; handles transposition via index mapping.
 */
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

/**
 * Backward pass for batched matrix multiplication w.r.t. matrix A.
 *
 * # Arguments
 * * `grad_out` – Gradient of the output batch [batch, m, n].
 * * `b` – Right batch [batch, k, n] (or transposed).
 * * `grad_a` – Gradient to accumulate into [batch, m, k].
 * * `batch`, `m`, `k`, `n` – Dimensions.
 *
 * # Notes
 * Uses shared memory tiling and grid‑stride loops.
 */
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

/**
 * Backward pass for batched matrix multiplication w.r.t. matrix B.
 *
 * # Arguments
 * * `a` – Left batch [batch, m, k].
 * * `grad_out` – Gradient of the output batch [batch, m, n].
 * * `grad_b` – Gradient to accumulate into [batch, k, n].
 * * `batch`, `m`, `k`, `n` – Dimensions.
 *
 * # Notes
 * Uses shared memory tiling and grid‑stride loops.
 */
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

/**
 * Backward pass for batched matrix multiplication when B is transposed.
 *
 * # Arguments
 * * `grad_out` – Gradient of the output batch [batch, m, n].
 * * `b` – Right batch stored transposed [batch, n, k].
 * * `grad_a` – Gradient to accumulate into [batch, m, k].
 * * `batch`, `m`, `k`, `n` – Dimensions.
 *
 * # Notes
 * Uses shared memory tiling and grid‑stride loops.
 */
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

/**
 * Backward pass for batched matrix multiplication when B is transposed, w.r.t. B.
 *
 * # Arguments
 * * `a` – Left batch [batch, m, k].
 * * `grad_out` – Gradient of the output batch [batch, m, n].
 * * `grad_b` – Gradient to accumulate into [batch, n, k] (transposed layout).
 * * `batch`, `m`, `k`, `n` – Dimensions.
 *
 * # Notes
 * Uses shared memory tiling and grid‑stride loops.
 */
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

/**
 * Gathers rows from a weight matrix using integer indices.
 *
 * # Arguments
 * * `weights` – Weight matrix [num_weights, hidden_dim].
 * * `indices` – Integer indices selecting rows, cast to float.
 * * `out` – Gathered output [out_size].
 * * `hidden_dim`, `out_size` – Dimensions.
 *
 * # Notes
 * Performs BF16 conversion on gathered values.
 */
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

/**
 * Backward pass for gather operation.
 *
 * # Arguments
 * * `indices` – Same indices as in forward.
 * * `grad_out` – Gradient of the gathered output.
 * * `grad_weights` – Gradient to accumulate into the weight matrix.
 * * `hidden_dim`, `out_size` – Dimensions.
 *
 * # Notes
 * Uses atomic adds; each gradient is added to the corresponding weight row.
 */
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

/**
 * RMS‑normalization of a batch of vectors.
 *
 * # Arguments
 * * `x` – Input tensor [num_vecs, dim].
 * * `w` – Scale parameters per dimension.
 * * `out` – Normalized output.
 * * `dim`, `eps`, `num_vecs` – Dimension, epsilon, and batch size.
 *
 * # Notes
 * Implements `x * w * sqrt(1 / (mean(x^2) + eps))` with BF16 conversion.
 */
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

/**
 * Backward pass for RMS‑normalization.
 *
 * # Arguments
 * * `x`, `w`, `grad_out` – Inputs and gradient of output.
 * * `grad_x`, `grad_w` – Gradients to accumulate into.
 * * `dim`, `eps`, `num_vecs` – Dimension, epsilon, and batch size.
 *
 * # Notes
 * Uses atomic adds for `grad_w` and BF16 conversion.
 */
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

/**
 * Cross‑entropy loss for a batch of logits and targets.
 *
 * # Arguments
 * * `logits` – Logit matrix [batch_size, num_classes] in BF16 format.
 * * `targets` – Target indices.
 * * `out_loss` – Scalar loss output.
 * * `batch_size`, `num_classes` – Dimensions.
 *
 * # Notes
 * Computes softmax and loss in a numerically stable manner.
 */
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

/**
 * Backward pass for cross‑entropy loss.
 *
 * # Arguments
 * * `logits`, `targets` – Inputs.
 * * `grad_out` – Gradient of the loss scalar.
 * * `grad_logits` – Gradient to accumulate into logits.
 * * `batch_size`, `num_classes` – Dimensions.
 *
 * # Notes
 * Computes gradient of softmax cross‑entropy w.r.t. logits.
 */
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

/**
 * Softmax activation over the last dimension of a batch.
 *
 * # Arguments
 * * `x` – Input tensor [B, N].
 * * `out` – Softmax output tensor.
 * * `B`, `N` – Batch size and dimension.
 *
 * # Notes
 * Uses a numerically stable algorithm with BF16 conversion.
 */
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

/**
 * Backward pass for softmax.
 *
 * # Arguments
 * * `out` – Forward softmax output.
 * * `grad_out` – Gradient of the loss w.r.t. softmax output.
 * * `grad_x` – Gradient to accumulate into input.
 * * `B`, `N` – Batch size and dimension.
 *
 * # Notes
 * Accumulates using atomic adds; BF16 conversion is applied.
 */
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

/**
 * Flash attention kernel for multi‑head attention.
 *
 * # Arguments
 * * `q`, `k`, `v` – Query, key, and value tensors.
 * * `out` – Output tensor.
 * * `batch`, `m`, `n`, `d` – Batch size, query length, key/value length, head dimension.
 * * `scale` – Scaling factor for dot‑products.
 * * `causal` – Whether to apply a causal mask.
 *
 * # Notes
 * Implements the efficient softmax‑based attention with numerically stable accumulation.
 */
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

/**
 * Transposes a 4‑D tensor from layout B‑S‑H‑D to B‑H‑S‑D.
 *
 * # Arguments
 * * `src` – Source tensor.
 * * `dst` – Destination tensor.
 * * `B`, `S`, `H`, `D` – Dimensions of the source tensor.
 *
 * # Notes
 * Uses a single thread per element; BF16 conversion is applied.
 */
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

/**
 * Backward pass for the 0213 transpose.
 *
 * # Arguments
 * * `grad_out` – Gradient of the output tensor.
 * * `grad_src` – Gradient to accumulate into the source tensor.
 * * `B`, `S`, `H`, `D` – Dimensions.
 *
 * # Notes
 * Accumulates gradient via atomic adds with BF16 conversion.
 */
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

/**
 * Rotary positional encoding (ROPE) for query/key tensors.
 *
 * # Arguments
 * * `x` – Input tensor.
 * * `out` – Rotated tensor.
 * * `seq_len`, `hidden_dim`, `head_dim`, `num_pairs` – Layout parameters.
 *
 * # Notes
 * Applies sine/cosine rotations per head dimension.
 */
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

/**
 * Backward pass for ROPE.
 *
 * # Arguments
 * * `grad_out` – Gradient of the rotated tensor.
 * * `grad_x` – Gradient to accumulate into the original tensor.
 * * `seq_len`, `hidden_dim`, `head_dim`, `num_pairs` – Layout parameters.
 *
 * # Notes
 * Uses the inverse rotation to propagate gradients.
 */
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

/**
 * AdamW optimizer step for a single weight vector.
 *
 * # Arguments
 * * `weights` – Parameters to update.
 * * `grads` – Gradients.
 * * `m`, `v` – First‑ and second‑moment estimates.
 * * `lr`, `beta1`, `beta2`, `eps`, `weight_decay` – Hyper‑parameters.
 * * `bc1`, `bc2` – Bias‑correction factors.
 * * `n` – Length of the weight vector.
 *
 * # Notes
 * Updates weights in place with BF16 conversion.
 */
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


/**
 * Repeats KV heads to match the number of query heads.
 *
 * # Arguments
 * * `input` – KV tensor [batch, kv_heads, seq_len, head_dim].
 * * `output` – Repeated tensor [batch, q_heads, seq_len, head_dim].
 * * `batch`, `num_kv_heads`, `repeats`, `seq_len`, `head_dim` – Layout parameters.
 *
 * # Notes
 * Performs a simple index mapping; no atomic operations needed.
 */
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

/**
 * Backward pass for repeat‑KV operation.
 *
 * # Arguments
 * * `grad_out` – Gradient of the repeated tensor.
 * * `grad_in` – Gradient to accumulate into the KV tensor.
 * * `batch`, `num_kv_heads`, `repeats`, `seq_len`, `head_dim` – Layout parameters.
 *
 * # Notes
 * Sums gradients across repeated heads using a loop.
 */
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
/**
 * Transposed matrix multiplication: C = A × Bᵀ.
 *
 * # Arguments
 * * `A` – Matrix [M, K].
 * * `B` – Matrix [N, K] (to be transposed).
 * * `C` – Result matrix [M, N].
 * * `M`, `K`, `N` – Dimensions.
 *
 * # Notes
 * Uses straightforward per‑element multiplication; no shared memory.
 */
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
/**
 * Backward pass for transposed matrix multiplication w.r.t. Aᵀ.
 *
 * # Arguments
 * * `dC` – Gradient of the output matrix [M, N].
 * * `A` – Matrix [M, K].
 * * `dB` – Gradient to accumulate into Bᵀ [N, K].
 * * `M`, `N`, `K` – Dimensions.
 *
 * # Notes
 * Accumulates gradients via atomic adds.
 */
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
