// ─────────────────────────────────────────────────────────────
// RapidaDB — Distance Kernels
//
// Implements cosine similarity, L2 distance, and dot product
// with optimized and naive versions.
// ─────────────────────────────────────────────────────────────

#include <torch/torch.h>
#include <cuda_runtime.h>
#include "rapidadb/kernels/distance.h"
#include "rapidadb/core/cuda_utils.h"

namespace rapidadb {

// ─── Optimized L2 Distance Kernel (Shared Memory + Vectorized) ───

constexpr int TILE_DIM = 128;
constexpr int THREADS_PER_BLOCK = 256;

template<int DIM>
__global__ void l2_distance_kernel_optimized(
    const float* __restrict__ queries,
    const float* __restrict__ database,
    float* __restrict__ output,
    int batch_size,
    int num_vectors,
    int dim
) {
    __shared__ float smem_query[TILE_DIM];
    
    int query_idx = blockIdx.x;
    int db_idx = blockIdx.y * blockDim.x + threadIdx.x;
    
    if (query_idx >= batch_size) return;
    
    const float* q = queries + query_idx * dim;
    
    // Load query vector into shared memory cooperatively
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        smem_query[i] = q[i];
    }
    __syncthreads();
    
    if (db_idx >= num_vectors) return;
    
    const float* d = database + db_idx * dim;
    float dist = 0.0f;
    
    // Vectorized computation using float4 when possible
    if (dim % 4 == 0) {
        const float4* d4 = reinterpret_cast<const float4*>(d);
        int dim4 = dim / 4;
        
        for (int i = 0; i < dim4; i++) {
            float4 d_vec = d4[i];
            int base_idx = i * 4;
            
            float diff0 = smem_query[base_idx + 0] - d_vec.x;
            float diff1 = smem_query[base_idx + 1] - d_vec.y;
            float diff2 = smem_query[base_idx + 2] - d_vec.z;
            float diff3 = smem_query[base_idx + 3] - d_vec.w;
            
            dist += diff0 * diff0 + diff1 * diff1 + diff2 * diff2 + diff3 * diff3;
        }
    } else {
        // Fallback for non-multiple-of-4 dimensions
        for (int i = 0; i < dim; i++) {
            float diff = smem_query[i] - d[i];
            dist += diff * diff;
        }
    }
    
    output[query_idx * num_vectors + db_idx] = dist;
}

// ─── Optimized Dot Product Kernel (Shared Memory + Vectorized) ───

template<int DIM>
__global__ void dot_product_kernel_optimized(
    const float* __restrict__ queries,
    const float* __restrict__ database,
    float* __restrict__ output,
    int batch_size,
    int num_vectors,
    int dim
) {
    __shared__ float smem_query[TILE_DIM];
    
    int query_idx = blockIdx.x;
    int db_idx = blockIdx.y * blockDim.x + threadIdx.x;
    
    if (query_idx >= batch_size) return;
    
    const float* q = queries + query_idx * dim;
    
    // Load query into shared memory
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        smem_query[i] = q[i];
    }
    __syncthreads();
    
    if (db_idx >= num_vectors) return;
    
    const float* d = database + db_idx * dim;
    float dot = 0.0f;
    
    // Vectorized computation
    if (dim % 4 == 0) {
        const float4* d4 = reinterpret_cast<const float4*>(d);
        int dim4 = dim / 4;
        
        for (int i = 0; i < dim4; i++) {
            float4 d_vec = d4[i];
            int base_idx = i * 4;
            
            dot += smem_query[base_idx + 0] * d_vec.x;
            dot += smem_query[base_idx + 1] * d_vec.y;
            dot += smem_query[base_idx + 2] * d_vec.z;
            dot += smem_query[base_idx + 3] * d_vec.w;
        }
    } else {
        for (int i = 0; i < dim; i++) {
            dot += smem_query[i] * d[i];
        }
    }
    
    output[query_idx * num_vectors + db_idx] = dot;
}

// ─── Optimized Cosine Similarity Kernel ───

template<int DIM>
__global__ void cosine_similarity_kernel_optimized(
    const float* __restrict__ queries,
    const float* __restrict__ database,
    float* __restrict__ output,
    int batch_size,
    int num_vectors,
    int dim
) {
    __shared__ float smem_query[TILE_DIM];
    __shared__ float smem_query_norm;
    
    int query_idx = blockIdx.x;
    int db_idx = blockIdx.y * blockDim.x + threadIdx.x;
    
    if (query_idx >= batch_size) return;
    
    const float* q = queries + query_idx * dim;
    
    // Load query into shared memory and compute norm
    float local_q_norm = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float val = q[i];
        smem_query[i] = val;
        local_q_norm += val * val;
    }
    
    // Reduce query norm across threads
    __shared__ float norm_reduce[THREADS_PER_BLOCK];
    norm_reduce[threadIdx.x] = local_q_norm;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            norm_reduce[threadIdx.x] += norm_reduce[threadIdx.x + stride];
        }
        __syncthreads();
    }
    
    if (threadIdx.x == 0) {
        smem_query_norm = sqrtf(norm_reduce[0] + 1e-8f);
    }
    __syncthreads();
    
    if (db_idx >= num_vectors) return;
    
    const float* d = database + db_idx * dim;
    float dot = 0.0f;
    float d_norm = 0.0f;
    
    // Vectorized computation
    if (dim % 4 == 0) {
        const float4* d4 = reinterpret_cast<const float4*>(d);
        int dim4 = dim / 4;
        
        for (int i = 0; i < dim4; i++) {
            float4 d_vec = d4[i];
            int base_idx = i * 4;
            
            dot += smem_query[base_idx + 0] * d_vec.x;
            dot += smem_query[base_idx + 1] * d_vec.y;
            dot += smem_query[base_idx + 2] * d_vec.z;
            dot += smem_query[base_idx + 3] * d_vec.w;
            
            d_norm += d_vec.x * d_vec.x + d_vec.y * d_vec.y + 
                      d_vec.z * d_vec.z + d_vec.w * d_vec.w;
        }
    } else {
        for (int i = 0; i < dim; i++) {
            float di = d[i];
            dot += smem_query[i] * di;
            d_norm += di * di;
        }
    }
    
    float denom = smem_query_norm * sqrtf(d_norm + 1e-8f);
    output[query_idx * num_vectors + db_idx] = dot / denom;
}

// ─── Cosine Similarity Kernel (Naive) ───────────────────────
//
// Grid:  (batch_size, ceil(num_vectors / BLOCK_SIZE))
// Block: (BLOCK_SIZE)
//
// Each thread computes the cosine similarity between one query
// and one database vector.

__global__ void cosine_similarity_kernel_naive(
    const float* __restrict__ queries,     // [batch_size, dim]
    const float* __restrict__ database,    // [num_vectors, dim]
    float* __restrict__ output,            // [batch_size, num_vectors]
    int batch_size,
    int num_vectors,
    int dim
) {
    int query_idx = blockIdx.x;
    int db_idx = blockIdx.y * blockDim.x + threadIdx.x;

    if (query_idx >= batch_size || db_idx >= num_vectors) return;

    const float* q = queries + query_idx * dim;
    const float* d = database + db_idx * dim;

    float dot = 0.0f;
    float q_norm = 0.0f;
    float d_norm = 0.0f;

    for (int i = 0; i < dim; i++) {
        float qi = q[i];
        float di = d[i];
        dot += qi * di;
        q_norm += qi * qi;
        d_norm += di * di;
    }

    float denom = sqrtf(q_norm) * sqrtf(d_norm) + 1e-8f;
    output[query_idx * num_vectors + db_idx] = dot / denom;
}


// ─── L2 Distance Kernel (Naive) ─────────────────────────────

__global__ void l2_distance_kernel_naive(
    const float* __restrict__ queries,
    const float* __restrict__ database,
    float* __restrict__ output,
    int batch_size,
    int num_vectors,
    int dim
) {
    int query_idx = blockIdx.x;
    int db_idx = blockIdx.y * blockDim.x + threadIdx.x;

    if (query_idx >= batch_size || db_idx >= num_vectors) return;

    const float* q = queries + query_idx * dim;
    const float* d = database + db_idx * dim;

    float dist = 0.0f;

    for (int i = 0; i < dim; i++) {
        float diff = q[i] - d[i];
        dist += diff * diff;
    }

    output[query_idx * num_vectors + db_idx] = dist;
}


// ─── Dot Product Kernel (Naive) ─────────────────────────────

__global__ void dot_product_kernel_naive(
    const float* __restrict__ queries,
    const float* __restrict__ database,
    float* __restrict__ output,
    int batch_size,
    int num_vectors,
    int dim
) {
    int query_idx = blockIdx.x;
    int db_idx = blockIdx.y * blockDim.x + threadIdx.x;

    if (query_idx >= batch_size || db_idx >= num_vectors) return;

    const float* q = queries + query_idx * dim;
    const float* d = database + db_idx * dim;

    float dot = 0.0f;

    for (int i = 0; i < dim; i++) {
        dot += q[i] * d[i];
    }

    output[query_idx * num_vectors + db_idx] = dot;
}


// ─── Host-Side Wrappers ─────────────────────────────────────

torch::Tensor cosine_similarity_cuda(
    const torch::Tensor& queries,
    const torch::Tensor& database
) {
    TORCH_CHECK(queries.is_cuda(), "queries must be a CUDA tensor");
    TORCH_CHECK(database.is_cuda(), "database must be a CUDA tensor");
    TORCH_CHECK(queries.dim() == 2, "queries must be 2D [batch, dim]");
    TORCH_CHECK(database.dim() == 2, "database must be 2D [n, dim]");
    TORCH_CHECK(queries.size(1) == database.size(1), "dimension mismatch");
    TORCH_CHECK(queries.scalar_type() == torch::kFloat32, "queries must be float32");
    TORCH_CHECK(database.scalar_type() == torch::kFloat32, "database must be float32");

    int batch_size = queries.size(0);
    int num_vectors = database.size(0);
    int dim = queries.size(1);

    auto output = torch::empty({batch_size, num_vectors},
                               queries.options());

    dim3 grid(batch_size, CDIV(num_vectors, THREADS_PER_BLOCK));
    dim3 block(THREADS_PER_BLOCK);

    if (dim <= TILE_DIM) {
        cosine_similarity_kernel_optimized<TILE_DIM><<<grid, block>>>(
            queries.data_ptr<float>(),
            database.data_ptr<float>(),
            output.data_ptr<float>(),
            batch_size,
            num_vectors,
            dim
        );
    } else {
        cosine_similarity_kernel_naive<<<grid, block>>>(
            queries.data_ptr<float>(),
            database.data_ptr<float>(),
            output.data_ptr<float>(),
            batch_size,
            num_vectors,
            dim
        );
    }
    CUDA_KERNEL_CHECK();

    return output;
}

torch::Tensor l2_distance_cuda(
    const torch::Tensor& queries,
    const torch::Tensor& database
) {
    TORCH_CHECK(queries.is_cuda(), "queries must be a CUDA tensor");
    TORCH_CHECK(database.is_cuda(), "database must be a CUDA tensor");
    TORCH_CHECK(queries.dim() == 2, "queries must be 2D [batch, dim]");
    TORCH_CHECK(database.dim() == 2, "database must be 2D [n, dim]");
    TORCH_CHECK(queries.size(1) == database.size(1), "dimension mismatch");
    TORCH_CHECK(queries.scalar_type() == torch::kFloat32, "queries must be float32");
    TORCH_CHECK(database.scalar_type() == torch::kFloat32, "database must be float32");

    int batch_size = queries.size(0);
    int num_vectors = database.size(0);
    int dim = queries.size(1);

    auto output = torch::empty({batch_size, num_vectors},
                               queries.options());

    dim3 grid(batch_size, CDIV(num_vectors, THREADS_PER_BLOCK));
    dim3 block(THREADS_PER_BLOCK);

    // Use optimized kernel for dimensions <= 128 (fits in shared memory)
    if (dim <= TILE_DIM) {
        l2_distance_kernel_optimized<TILE_DIM><<<grid, block>>>(
            queries.data_ptr<float>(),
            database.data_ptr<float>(),
            output.data_ptr<float>(),
            batch_size,
            num_vectors,
            dim
        );
    } else {
        l2_distance_kernel_naive<<<grid, block>>>(
            queries.data_ptr<float>(),
            database.data_ptr<float>(),
            output.data_ptr<float>(),
            batch_size,
            num_vectors,
            dim
        );
    }
    CUDA_KERNEL_CHECK();

    return output;
}

torch::Tensor dot_product_cuda(
    const torch::Tensor& queries,
    const torch::Tensor& database
) {
    TORCH_CHECK(queries.is_cuda(), "queries must be a CUDA tensor");
    TORCH_CHECK(database.is_cuda(), "database must be a CUDA tensor");
    TORCH_CHECK(queries.dim() == 2, "queries must be 2D [batch, dim]");
    TORCH_CHECK(database.dim() == 2, "database must be 2D [n, dim]");
    TORCH_CHECK(queries.size(1) == database.size(1), "dimension mismatch");
    TORCH_CHECK(queries.scalar_type() == torch::kFloat32, "queries must be float32");
    TORCH_CHECK(database.scalar_type() == torch::kFloat32, "database must be float32");

    int batch_size = queries.size(0);
    int num_vectors = database.size(0);
    int dim = queries.size(1);

    auto output = torch::empty({batch_size, num_vectors},
                               queries.options());

    dim3 grid(batch_size, CDIV(num_vectors, THREADS_PER_BLOCK));
    dim3 block(THREADS_PER_BLOCK);

    if (dim <= TILE_DIM) {
        dot_product_kernel_optimized<TILE_DIM><<<grid, block>>>(
            queries.data_ptr<float>(),
            database.data_ptr<float>(),
            output.data_ptr<float>(),
            batch_size,
            num_vectors,
            dim
        );
    } else {
        dot_product_kernel_naive<<<grid, block>>>(
            queries.data_ptr<float>(),
            database.data_ptr<float>(),
            output.data_ptr<float>(),
            batch_size,
            num_vectors,
            dim
        );
    }
    CUDA_KERNEL_CHECK();

    return output;
}

}  // namespace rapidadb
