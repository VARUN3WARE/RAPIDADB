#include "rapidadb/clustering/kmeans.h"
#include "rapidadb/core/cuda_utils.h"
#include <vector>

namespace rapidadb {

// Forward declarations from kmeans.cu
void assign_clusters_cuda(
    const float* points, const float* centroids,
    int* assignments, float* distances,
    int n, int k, int d
);

void update_centroids_cuda(
    const float* points, const int* assignments,
    float* new_centroids, int* counts,
    int n, int k, int d
);

KMeansResult kmeans_cuda(
    const torch::Tensor& points,
    int k,
    int max_iters,
    float tol,
    int seed
) {
    // Memory-safe KMeans - won't crash your 4GB GPU
    int n = points.size(0);
    int d = points.size(1);
    
    auto options = points.options();
    auto centroids = torch::randn({k, d}, options);
    auto labels = torch::zeros({n}, options.dtype(torch::kInt32));
    auto distances = torch::zeros({n}, options);
    
    float prev_inertia = INFINITY;
    int iter = 0;
    
    for (iter = 0; iter < max_iters; ++iter) {
        // Assignment step
        assign_clusters_cuda(
            points.data_ptr<float>(),
            centroids.data_ptr<float>(),
            labels.data_ptr<int>(),
            distances.data_ptr<float>(),
            n, k, d
        );
        
        float inertia = distances.sum().item<float>();
        
        if (std::abs(prev_inertia - inertia) < tol) {
            break;  // Converged
        }
        prev_inertia = inertia;
        
        // Update step
        auto counts = torch::zeros({k}, options.dtype(torch::kInt32));
        update_centroids_cuda(
            points.data_ptr<float>(),
            labels.data_ptr<int>(),
            centroids.data_ptr<float>(),
            counts.data_ptr<int>(),
            n, k, d
        );
    }
    
    return {centroids, labels, torch::tensor(prev_inertia), iter};
}

}  // namespace rapidadb
