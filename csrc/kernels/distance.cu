// ─────────────────────────────────────────────────────────────
// RapidaDB — Distance Kernels (Naive, Week 1)
//
// Implements cosine similarity, L2 distance, and dot product
// as straightforward CUDA kernels. No shared memory tiling or
// vectorized loads yet — those come in Week 2.
// ─────────────────────────────────────────────────────────────

#include <torch/torch.h>
#include <cuda_runtime.h>
#include "rapidadb/kernels/distance.h"
#include "rapidadb/core/cuda_utils.h"

namespace rapidadb {

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

    dim3 grid(batch_size, CDIV(num_vectors, DEFAULT_BLOCK_SIZE));
    dim3 block(DEFAULT_BLOCK_SIZE);

    cosine_similarity_kernel_naive<<<grid, block>>>(
        queries.data_ptr<float>(),
        database.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        num_vectors,
        dim
    );
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

    dim3 grid(batch_size, CDIV(num_vectors, DEFAULT_BLOCK_SIZE));
    dim3 block(DEFAULT_BLOCK_SIZE);

    l2_distance_kernel_naive<<<grid, block>>>(
        queries.data_ptr<float>(),
        database.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        num_vectors,
        dim
    );
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

    dim3 grid(batch_size, CDIV(num_vectors, DEFAULT_BLOCK_SIZE));
    dim3 block(DEFAULT_BLOCK_SIZE);

    dot_product_kernel_naive<<<grid, block>>>(
        queries.data_ptr<float>(),
        database.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        num_vectors,
        dim
    );
    CUDA_KERNEL_CHECK();

    return output;
}

}  // namespace rapidadb
