#pragma once

#include <torch/torch.h>
#include "rapidadb/core/types.h"

namespace rapidadb {

/// Select top-k smallest values (for L2) or largest values (for cosine/dot)
/// from a distance matrix.
///
/// Args:
///   distances — [batch_size, num_vectors], float32, CUDA
///   k         — number of top elements to select
///   largest   — if true, select largest (cosine/dot); if false, select smallest (L2)
///
/// Returns:
///   SearchResult with:
///     distances — [batch_size, k], float32
///     indices   — [batch_size, k], int64

// Baseline: Thrust/CUB (Week 1)
SearchResult topk_thrust(
    const torch::Tensor& distances,
    int k,
    bool largest = true
);

// Week 3: Custom warp-level heap (k <= 128)
SearchResult topk_warp_heap(
    const torch::Tensor& distances,
    int k,
    bool largest = true
);

// Week 3: Auto-select best algorithm
SearchResult topk_auto(
    const torch::Tensor& distances,
    int k,
    bool largest = true
);

}  // namespace rapidadb
