#pragma once

#include <torch/torch.h>
#include "rapidadb/kernels/topk.h"

namespace rapidadb {

/// Fused L2 distance + top-k selection
/// Computes distances and finds top-k in single kernel pass
/// No intermediate buffer - saves memory and bandwidth
SearchResult fused_l2_topk(
    const torch::Tensor& queries,
    const torch::Tensor& database,
    int k
);

}  // namespace rapidadb
