#pragma once

#include <torch/torch.h>
#include "rapidadb/kernels/topk.h"

namespace rapidadb {

// Fused L2 distance + top-k kernel
// No intermediate matrix - compute distances on-the-fly
SearchResult l2_topk_fused(
    const torch::Tensor& queries,
    const torch::Tensor& database,
    int k
);

}  // namespace rapidadb
