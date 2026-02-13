#pragma once

#include <torch/torch.h>
#include <cstdint>

namespace rapidadb {

// ─── Precision Modes ─────────────────────────────────────────
enum class Precision {
    FP32,
    FP16,
    BF16,
    INT8,
};

// ─── Distance Metrics ────────────────────────────────────────
enum class Metric {
    COSINE,
    L2,
    DOT_PRODUCT,
};

// ─── Search Result ───────────────────────────────────────────
struct SearchResult {
    torch::Tensor distances;  // [batch_size, k]
    torch::Tensor indices;    // [batch_size, k]
};

// ─── Index Configuration ────────────────────────────────────
struct IndexConfig {
    int dim = 768;
    Metric metric = Metric::COSINE;
    Precision precision = Precision::FP32;

    // IVF parameters
    int nlist = 4096;
    int nprobe = 64;

    // PQ parameters
    int pq_m = 96;       // Number of subspaces
    int pq_k = 256;      // Centroids per subspace

    // Runtime
    int num_streams = 4;
    int max_batch_size = 128;
};

}  // namespace rapidadb
