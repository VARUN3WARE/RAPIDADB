#pragma once

#include <torch/torch.h>
#include "rapidadb/core/types.h"

namespace rapidadb {

/// FlatIndex: Brute-force exact nearest neighbor search.
///
/// Stores all vectors in GPU memory and computes exhaustive distances
/// for every search query. Guarantees 100% recall but O(n) per query.
class FlatIndex {
public:
    /// Construct an empty FlatIndex.
    ///
    /// @param dim       Dimensionality of vectors
    /// @param metric    Distance metric (COSINE, L2, DOT_PRODUCT)
    explicit FlatIndex(int dim, Metric metric = Metric::COSINE);

    /// Add vectors to the index.
    ///
    /// @param vectors   [n, dim] float32 tensor on CUDA
    /// @param ids       [n] int64 tensor (optional; auto-assigns if empty)
    void add(const torch::Tensor& vectors,
             const torch::Tensor& ids = torch::Tensor());

    /// Search for k nearest neighbors.
    ///
    /// @param queries   [batch_size, dim] float32 tensor on CUDA
    /// @param k         Number of nearest neighbors to return
    /// @return SearchResult with distances [batch, k] and indices [batch, k]
    SearchResult search(const torch::Tensor& queries, int k) const;

    /// Reset the index (remove all vectors).
    void reset();

    /// Number of vectors currently in the index.
    int64_t size() const { return num_vectors_; }

    /// Dimensionality of vectors.
    int dim() const { return dim_; }

    /// Distance metric.
    Metric metric() const { return metric_; }

private:
    int dim_;
    Metric metric_;
    int64_t num_vectors_ = 0;

    torch::Tensor vectors_;   // [num_vectors, dim], float32, CUDA
    torch::Tensor ids_;       // [num_vectors], int64, CUDA
};

}  // namespace rapidadb
