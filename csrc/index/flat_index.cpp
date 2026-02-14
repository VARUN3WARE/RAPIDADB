// ─────────────────────────────────────────────────────────────
// RapidaDB — FlatIndex Implementation
// ─────────────────────────────────────────────────────────────

#include "rapidadb/index/flat_index.h"
#include "rapidadb/kernels/distance.h"
#include "rapidadb/kernels/topk.h"
#include "rapidadb/core/cuda_utils.h"

namespace rapidadb {

FlatIndex::FlatIndex(int dim, Metric metric)
    : dim_(dim), metric_(metric), num_vectors_(0) {}

void FlatIndex::add(const torch::Tensor& vectors, const torch::Tensor& ids) {
    TORCH_CHECK(vectors.is_cuda(), "vectors must be on CUDA");
    TORCH_CHECK(vectors.dim() == 2 && vectors.size(1) == dim_,
                "vectors must be [n, " + std::to_string(dim_) + "]");
    TORCH_CHECK(vectors.scalar_type() == torch::kFloat32,
                "vectors must be float32");

    int64_t n = vectors.size(0);

    // Handle IDs
    torch::Tensor new_ids;
    if (ids.defined() && ids.numel() > 0) {
        TORCH_CHECK(ids.size(0) == n, "ids length must match vectors");
        new_ids = ids.to(torch::kInt64).to(vectors.device());
    } else {
        // Auto-assign sequential IDs
        new_ids = torch::arange(num_vectors_, num_vectors_ + n,
                                torch::TensorOptions()
                                    .dtype(torch::kInt64)
                                    .device(vectors.device()));
    }

    // Append to existing storage
    if (num_vectors_ == 0) {
        vectors_ = vectors.clone();
        ids_ = new_ids.clone();
    } else {
        vectors_ = torch::cat({vectors_, vectors}, /*dim=*/0);
        ids_ = torch::cat({ids_, new_ids}, /*dim=*/0);
    }

    num_vectors_ += n;
}

SearchResult FlatIndex::search(const torch::Tensor& queries, int k) const {
    TORCH_CHECK(queries.is_cuda(), "queries must be on CUDA");
    TORCH_CHECK(queries.dim() == 2 && queries.size(1) == dim_,
                "queries must be [batch, " + std::to_string(dim_) + "]");
    TORCH_CHECK(num_vectors_ > 0, "index is empty");
    TORCH_CHECK(k > 0 && k <= num_vectors_,
                "k must be in [1, " + std::to_string(num_vectors_) + "]");

    // Step 1: Compute all pairwise distances
    torch::Tensor distances;
    bool select_largest;

    switch (metric_) {
        case Metric::COSINE:
            distances = cosine_similarity_cuda(queries, vectors_);
            select_largest = true;   // Higher similarity = better
            break;
        case Metric::L2:
            distances = l2_distance_cuda(queries, vectors_);
            select_largest = false;  // Lower distance = better
            break;
        case Metric::DOT_PRODUCT:
            distances = dot_product_cuda(queries, vectors_);
            select_largest = true;   // Higher dot product = better
            break;
        default:
            TORCH_CHECK(false, "Unknown metric");
    }

    // Step 2: Select top-k
    auto result = topk_thrust(distances, k, select_largest);

    // Step 3: Map local indices to user IDs
    result.indices = ids_.index_select(0, result.indices.flatten())
                         .reshape(result.indices.sizes());

    return result;
}

void FlatIndex::reset() {
    vectors_ = torch::Tensor();
    ids_ = torch::Tensor();
    num_vectors_ = 0;
}

}  // namespace rapidadb
