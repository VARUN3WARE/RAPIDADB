#pragma once

#include <cuda_runtime.h>

namespace rapidadb {

// Warp-level reduction primitives for Week 3

constexpr int WARP_SIZE = 32;

// Warp-level sum reduction using shuffle
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Warp-level min reduction using shuffle
__device__ __forceinline__ float warp_reduce_min(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = fminf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// Warp-level max reduction using shuffle
__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// Block-level reduction using warp shuffle + shared memory between warps
template<int BLOCK_SIZE>
__device__ float block_reduce_sum(float val) {
    __shared__ float warp_sums[BLOCK_SIZE / WARP_SIZE];
    
    int lane_id = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    
    // Reduce within warp
    val = warp_reduce_sum(val);
    
    // First thread in each warp writes to shared memory
    if (lane_id == 0) {
        warp_sums[warp_id] = val;
    }
    __syncthreads();
    
    // First warp reduces across warps
    if (warp_id == 0) {
        val = (lane_id < (BLOCK_SIZE / WARP_SIZE)) ? warp_sums[lane_id] : 0.0f;
        val = warp_reduce_sum(val);
    }
    
    return val;
}

}  // namespace rapidadb
