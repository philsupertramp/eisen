// ============================================================
// SHARED MEMORY TILE SIZE.
// Must match block_dim (16, 16, 1) used in graph.rs launch configs.
// Threads cooperatively load a TILE_SIZE x TILE_SIZE block into
// fast SRAM, then each thread's inner loop reads from SRAM instead
// of global VRAM, reducing bandwidth by ~TILE_SIZE.
// ============================================================
#define TILE_SIZE 16

#ifdef USE_BF16_ARITH
#include <cuda_bf16.h>
__device__ __forceinline__ float bf16q(float x) {
    return __bfloat162float(__float2bfloat16(x));
}
#else
__device__ __forceinline__ float bf16q(float x) { return x; }
#endif


