// Fused distance + top-k kernel
// Why compute all distances when we only need k? That's like reading the whole menu when you know you want pizza.

#include <cuda_runtime.h>
#include "rapidadb/kernels/search_fused.h"
#include "rapidadb/kernels/warp_utils.h"
#include "rapidadb/core/cuda_utils.h"

namespace rapidadb {

// Warp-level min heap for tracking top-k smallest distances
template<int K>
struct WarpMinHeap {
    float distances[K];
    int indices[K];
    
    __device__ void init() {
        #pragma unroll
        for (int i = 0; i < K; ++i) {
            distances[i] = FLT_MAX;
            indices[i] = -1;
        }
    }
    
    __device__ void insert(float dist, int idx) {
        // Simple insertion - if dist smaller than max, replace max
        if (dist < distances[K-1]) {
            distances[K-1] = dist;
            indices[K-1] = idx;
            
            // Bubble down to maintain heap property (simple version)
            for (int i = K-2; i >= 0; --i) {
                if (distances[i] > distances[i+1]) {
                    float tmp_d = distances[i];
                    int tmp_i = indices[i];
                    distances[i] = distances[i+1];
                    indices[i] = indices[i+1];
                    distances[i+1] = tmp_d;
                    indices[i+1] = tmp_i;
                }
            }
        }
    }
};

// Fused L2 distance + top-k kernel with register tiling
template<int K, int TILE_SIZE = 8>
__global__ void l2_topk_fused_kernel(
    const float* __restrict__ queries,
    const float* __restrict__ database,
    float* __restrict__ out_distances,
    int* __restrict__ out_indices,
    int batch_size,
    int num_vectors,
    int dim
) {
    const int query_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    
    if (query_idx >= batch_size) return;
    
    // Each thread maintains its own heap
    WarpMinHeap<K> heap;
    heap.init();
    
    const float* query = queries + query_idx * dim;
    
    // Process vectors in chunks
    for (int vec_base = tid; vec_base < num_vectors; vec_base += blockDim.x) {
        const float* db_vec = database + vec_base * dim;
        
        // Compute L2 distance with register tiling
        float dist = 0.0f;
        
        #pragma unroll 4
        for (int d = 0; d < dim; d += TILE_SIZE) {
            float reg_query[TILE_SIZE];
            float reg_db[TILE_SIZE];
            
            // Load tiles into registers
            #pragma unroll
            for (int i = 0; i < TILE_SIZE && (d + i) < dim; ++i) {
                reg_query[i] = query[d + i];
                reg_db[i] = db_vec[d + i];
            }
            
            // Compute partial distance
            #pragma unroll
            for (int i = 0; i < TILE_SIZE && (d + i) < dim; ++i) {
                float diff = reg_query[i] - reg_db[i];
                dist += diff * diff;
            }
        }
        
        heap.insert(dist, vec_base);
    }
    
    // Write results
    int out_offset = query_idx * K;
    if (tid < K) {
        out_distances[out_offset + tid] = heap.distances[tid];
        out_indices[out_offset + tid] = heap.indices[tid];
    }
}

SearchResult l2_topk_fused(
    const torch::Tensor& queries,
    const torch::Tensor& database,
    int k
) {
    CHECK_CUDA(queries);
    CHECK_CUDA(database);
    
    int batch_size = queries.size(0);
    int num_vectors = database.size(0);
    int dim = queries.size(1);
    
    auto distances = torch::empty({batch_size, k}, queries.options());
    auto indices = torch::empty({batch_size, k}, 
                                queries.options().dtype(torch::kInt32));
    
    dim3 grid(batch_size);
    dim3 block(256);  // Memory-bound, modest thread count
    
    // Template dispatch based on k
    if (k <= 10) {
        l2_topk_fused_kernel<10><<<grid, block>>>(
            queries.data_ptr<float>(),
            database.data_ptr<float>(),
            distances.data_ptr<float>(),
            indices.data_ptr<int>(),
            batch_size, num_vectors, dim
        );
    } else if (k <= 32) {
        l2_topk_fused_kernel<32><<<grid, block>>>(
            queries.data_ptr<float>(),
            database.data_ptr<float>(),
            distances.data_ptr<float>(),
            indices.data_ptr<int>(),
            batch_size, num_vectors, dim
        );
    } else {
        l2_topk_fused_kernel<128><<<grid, block>>>(
            queries.data_ptr<float>(),
            database.data_ptr<float>(),
            distances.data_ptr<float>(),
            indices.data_ptr<int>(),
            batch_size, num_vectors, dim
        );
    }
    
    CUDA_KERNEL_CHECK();
    
    return {distances, indices.to(torch::kInt64)};
}

}  // namespace rapidadb
