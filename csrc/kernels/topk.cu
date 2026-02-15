// ─────────────────────────────────────────────────────────────
// RapidaDB — Top-K Selection Kernels (Week 1: Thrust baseline)
//
// Initial implementation using torch::topk (which uses CUB/Thrust
// internally). Custom warp-heap and bitonic kernels come in Week 3.
// ─────────────────────────────────────────────────────────────

#include <torch/torch.h>
#include "rapidadb/kernels/topk.h"
#include "rapidadb/core/cuda_utils.h"

namespace rapidadb {

SearchResult topk_thrust(
    const torch::Tensor& distances,
    int k,
    bool largest
) {
    TORCH_CHECK(distances.is_cuda(), "distances must be a CUDA tensor");
    TORCH_CHECK(distances.dim() == 2, "distances must be 2D [batch, n]");
    TORCH_CHECK(k > 0 && k <= distances.size(1),
                "k must be in [1, num_vectors]");

    // torch::topk wraps CUB/Thrust internally — good baseline
    auto [values, indices] = torch::topk(distances, k, /*dim=*/1, largest);

    return SearchResult{values, indices};
}

}  // namespace rapidadb
