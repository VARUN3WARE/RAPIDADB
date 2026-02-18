// ─────────────────────────────────────────────────────────────
// RapidaDB — Warp-Level Top-K Kernels (Week 3)
//
// Custom top-k using per-warp min-heap in registers.
// No synchronization within warp (lockless).
// Target: k <= 128 for optimal performance.
// ─────────────────────────────────────────────────────────────

#include <torch/torch.h>
#include <cuda_runtime.h>
#include "rapidadb/kernels/topk.h"
#include "rapidadb/kernels/warp_utils.h"
#include "rapidadb/core/cuda_utils.h"

namespace rapidadb {

// Warp-level heap top-k kernel
// Each warp maintains a min-heap of k elements in registers
template<int K>
__global__ void topk_warp_heap_kernel(
    const float* distances,
    float* out_distances,
    int64_t* out_indices,
    int batch_size,
    int num_vectors
) {
    int batch_idx = blockIdx.x;
    int lane_id = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    
    if (batch_idx >= batch_size) return;
    
    const float* batch_distances = distances + batch_idx * num_vectors;
    
    // Per-thread heap (each thread maintains K/WARP_SIZE elements)
    constexpr int ELEMS_PER_THREAD = (K + WARP_SIZE - 1) / WARP_SIZE;
    float heap_vals[ELEMS_PER_THREAD];
    int heap_idxs[ELEMS_PER_THREAD];
    
    // Initialize heap with -inf
    #pragma unroll
    for (int i = 0; i < ELEMS_PER_THREAD; i++) {
        heap_vals[i] = -INFINITY;
        heap_idxs[i] = -1;
    }
    
    // Process elements
    for (int i = lane_id + warp_id * WARP_SIZE; i < num_vectors; i += blockDim.x) {
        float val = batch_distances[i];
        
        // Simple insertion: find max in heap and replace if val is smaller
        int max_idx = 0;
        float max_val = heap_vals[0];
        #pragma unroll
        for (int j = 1; j < ELEMS_PER_THREAD; j++) {
            if (heap_vals[j] > max_val) {
                max_val = heap_vals[j];
                max_idx = j;
            }
        }
        
        if (val < max_val) {
            heap_vals[max_idx] = val;
            heap_idxs[max_idx] = i;
        }
    }
    
    // Write results (simplified - needs proper merging across threads)
    if (warp_id == 0 && lane_id < K) {
        int thread_id = lane_id % WARP_SIZE;
        int elem_id = lane_id / WARP_SIZE;
        if (elem_id < ELEMS_PER_THREAD) {
            out_distances[batch_idx * K + lane_id] = heap_vals[elem_id];
            out_indices[batch_idx * K + lane_id] = heap_idxs[elem_id];
        }
    }
}

// Host wrapper for warp-level heap top-k
SearchResult topk_warp_heap(
    const torch::Tensor& distances,
    int k,
    bool largest
) {
    TORCH_CHECK(distances.is_cuda(), "distances must be a CUDA tensor");
    TORCH_CHECK(distances.dim() == 2, "distances must be 2D [batch, n]");
    TORCH_CHECK(k > 0 && k <= 128, "warp heap supports k in [1, 128]");
    
    int batch_size = distances.size(0);
    int num_vectors = distances.size(1);
    
    auto out_distances = torch::empty({batch_size, k}, distances.options());
    auto out_indices = torch::empty({batch_size, k}, 
                                    torch::TensorOptions().dtype(torch::kInt64).device(distances.device()));
    
    // For now, use thrust as fallback
    // TODO: Implement proper warp-level merge and sorting
    return topk_thrust(distances, k, largest);
}

// Auto-select best top-k algorithm
SearchResult topk_auto(
    const torch::Tensor& distances,
    int k,
    bool largest
) {
    // For now, always use thrust
    // TODO: Add heuristics based on k and num_vectors
    return topk_thrust(distances, k, largest);
}

}  // namespace rapidadb
