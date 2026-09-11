/*!
 * @file common.cuh
 * @brief Shared CUDA utilities for the Eisen kernels.
 *
 * This header defines constants and helper functions that are used
 * across the various CUDA kernels in the project.  The constants
 * are tuned to match the block dimensions defined in the Rust
 * launch configuration (`graph.rs`).
 */
// ============================================================

// Must match block_dim (16, 16, 1) used in graph.rs launch configs.
// Threads cooperatively load a TILE_SIZE x TILE_SIZE block into
// fast SRAM, then each thread's inner loop reads from SRAM instead
// of global VRAM, reducing bandwidth by ~TILE_SIZE.
// ============================================================
/**
 * Shared memory tile size.
 *
 * Must match the `block_dim` (16, 16, 1) used in the launch
 * configurations in `graph.rs`.  Threads cooperatively load a
 * `TILE_SIZE × TILE_SIZE` block into fast SRAM, and then each
 * thread reads from SRAM instead of global VRAM in its inner loop.
 * This reduces bandwidth usage by a factor of roughly `TILE_SIZE`.
 */
#define TILE_SIZE 16

// ============================================================
// BF16 Arithmetic Helper
// ============================================================
#ifdef USE_BF16_ARITH
#include <cuda_bf16.h>
/**
 * Convert a FP32 value to BF16 and back, effectively casting to BF16.
 *
 * This helper is used when BF16 arithmetic is enabled via the
 * `USE_BF16_ARITH` compilation flag.  It takes a 32‑bit floating point
 * number, casts it to a bfloat16 representation using CUDA’s intrinsics
 * and then converts it back to `float`.  The result has the same
 * precision as a true BF16 cast, which can be useful for emulating
 * BF16 behaviour on hardware that only supports FP32.
 *
 * # Arguments
 * * `x` – The FP32 value to be cast.
 *
 * # Returns
 * The value cast to BF16 and back to FP32.
 */
__device__ __forceinline__ float bf16q(float x) {
    return __bfloat162float(__float2bfloat16(x));
}
#else
/**
 * Identity function when BF16 arithmetic is disabled.
 *
 * When `USE_BF16_ARITH` is not defined, the helper simply returns
 * the input unchanged.  This keeps the same API surface whether or
 * not BF16 support is compiled in.
 */
__device__ __forceinline__ float bf16q(float x) { return x; }
#endif


