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
