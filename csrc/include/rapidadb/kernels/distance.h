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

/// Compute cosine similarity between query batch and database vectors.
/// Returns similarity scores (higher = more similar).
torch::Tensor cosine_similarity_cuda(
    const torch::Tensor& queries,
    const torch::Tensor& database
);

/// Compute squared L2 distance between query batch and database vectors.
/// Returns distances (lower = more similar).
torch::Tensor l2_distance_cuda(
    const torch::Tensor& queries,
    const torch::Tensor& database
);

/// Compute dot product between query batch and database vectors.
/// Returns dot products (higher = more similar for normalized vectors).
torch::Tensor dot_product_cuda(
    const torch::Tensor& queries,
    const torch::Tensor& database
);

}  // namespace rapidadb
