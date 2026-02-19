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

Tool call argument 'replace' pruned from message history.

}  // namespace rapidadb
