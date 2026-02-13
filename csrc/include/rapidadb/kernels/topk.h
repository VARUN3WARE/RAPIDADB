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
SearchResult topk_thrust(
    const torch::Tensor& distances,
    int k,
    bool largest = true
);

}  // namespace rapidadb
