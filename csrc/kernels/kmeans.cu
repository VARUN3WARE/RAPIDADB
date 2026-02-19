// KMeans clustering on GPU - because CPUs think clustering is hard
// Spoiler: GPUs think it's Tuesday

#include <torch/torch.h>
#include <cuda_runtime.h>
#include "rapidadb/core/cuda_utils.h"

namespace rapidadb {

// Assign points to nearest centroids
__global__ void assign_clusters_kernel(
    const float* points,      // [n, d]
    const float* centroids,   // [k, d]
    int* assignments,         // [n]
    float* distances,         // [n] - distance to assigned centroid
    int n,
    int k,
    int d
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    const float* point = points + idx * d;
    float min_dist = INFINITY;
    int best_cluster = 0;

    // Find nearest centroid - the GPU way: try them all at once
    for (int c = 0; c < k; ++c) {
        const float* centroid = centroids + c * d;
        float dist = 0.0f;
        
        for (int i = 0; i < d; ++i) {
            float diff = point[i] - centroid[i];
            dist += diff * diff;
        }
        
        if (dist < min_dist) {
            min_dist = dist;
            best_cluster = c;
        }
    }
    
    assignments[idx] = best_cluster;
    distances[idx] = sqrtf(min_dist);
}

// Update centroids based on assignments
__global__ void update_centroids_kernel(
    const float* points,      // [n, d]
    const int* assignments,   // [n]
    float* new_centroids,     // [k, d]
    int* counts,              // [k]
    int n,
    int k,
    int d
) {
    int c = blockIdx.x;  // One block per centroid
    int tid = threadIdx.x;
    
    if (c >= k) return;
    
    extern __shared__ float shared_centroid[];
    
    // Initialize shared memory
    for (int i = tid; i < d; i += blockDim.x) {
        shared_centroid[i] = 0.0f;
    }
    __syncthreads();
    
    int count = 0;
    
    // Accumulate points assigned to this centroid
    for (int i = tid; i < n; i += blockDim.x) {
        if (assignments[i] == c) {
            count++;
            const float* point = points + i * d;
            for (int j = 0; j < d; ++j) {
                atomicAdd(&shared_centroid[j], point[j]);
            }
        }
    }
    
    // Reduce count across threads
    atomicAdd(&counts[c], count);
    __syncthreads();
    
    // Average and write to global memory
    int total_count = counts[c];
    if (total_count > 0) {
        for (int i = tid; i < d; i += blockDim.x) {
            new_centroids[c * d + i] = shared_centroid[i] / total_count;
        }
    }
}

}  // namespace rapidadb
