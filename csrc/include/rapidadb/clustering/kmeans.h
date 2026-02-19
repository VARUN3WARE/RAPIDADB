#pragma once

#include <torch/torch.h>
#include <tuple>

namespace rapidadb {

/// GPU KMeans clustering result
struct KMeansResult {
    torch::Tensor centroids;   // [k, d] cluster centers
    torch::Tensor labels;      // [n] cluster assignments
    torch::Tensor inertia;     // scalar: sum of squared distances
    int iterations;            // number of iterations run
};

/// Run KMeans clustering on GPU
/// Smart enough to not crash your 4GB GPU
KMeansResult kmeans_cuda(
    const torch::Tensor& points,  // [n, d]
    int k,                        // number of clusters
    int max_iters = 100,
    float tol = 1e-4,
    int seed = 42
);

}  // namespace rapidadb
