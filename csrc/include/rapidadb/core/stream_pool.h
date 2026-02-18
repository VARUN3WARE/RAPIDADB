#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <memory>

namespace rapidadb {

/// Multi-stream pool for overlapping compute and memory transfers
/// Why did the CUDA stream cross the road? To avoid synchronization!
class StreamPool {
public:
    explicit StreamPool(int num_streams = 4);
    ~StreamPool();

    // Get stream for async operations
    cudaStream_t get_stream(int idx);
    
    // Synchronize all streams (blocking)
    void synchronize_all();
    
    // Get number of streams
    int size() const { return streams_.size(); }

private:
    std::vector<cudaStream_t> streams_;
};

}  // namespace rapidadb
