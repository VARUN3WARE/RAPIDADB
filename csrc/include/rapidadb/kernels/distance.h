#pragma once

#include <torch/torch.h>

namespace rapidadb {

// ─── Distance Kernel Declarations ───────────────────────────
//
// All kernels operate on GPU tensors and return GPU tensors.
// Inputs:
//   queries  — [batch_size, dim], float32, CUDA
//   database — [num_vectors, dim], float32, CUDA
// Output:
//   distances — [batch_size, num_vectors], float32, CUDA

// ─── Naive Implementations (Week 1) ─────────────────────────

torch::Tensor cosine_similarity_cuda(
    const torch::Tensor& queries,
    const torch::Tensor& database
);

torch::Tensor l2_distance_cuda(
    const torch::Tensor& queries,
    const torch::Tensor& database
);

torch::Tensor dot_product_cuda(
    const torch::Tensor& queries,
    const torch::Tensor& database
);

// ─── Optimized Implementations (Week 2) ─────────────────────

torch::Tensor cosine_similarity_cuda_optimized(
    const torch::Tensor& queries,
    const torch::Tensor& database
);

torch::Tensor l2_distance_cuda_optimized(
    const torch::Tensor& queries,
    const torch::Tensor& database
);

torch::Tensor dot_product_cuda_optimized(
    const torch::Tensor& queries,
    const torch::Tensor& database
);

}  // namespace rapidadb
