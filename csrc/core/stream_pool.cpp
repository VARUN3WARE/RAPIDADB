#include "rapidadb/core/stream_pool.h"
#include "rapidadb/core/cuda_utils.h"
#include <stdexcept>

namespace rapidadb {

StreamPool::StreamPool(int num_streams) {
    // Parallel streams: because waiting is for CPUs
    streams_.resize(num_streams);
    for (int i = 0; i < num_streams; ++i) {
        CUDA_CHECK(cudaStreamCreate(&streams_[i]));
    }
}

StreamPool::~StreamPool() {
    for (auto stream : streams_) {
        cudaStreamDestroy(stream);  // No error checking in destructor
    }
}

cudaStream_t StreamPool::get_stream(int idx) {
    if (idx < 0 || idx >= static_cast<int>(streams_.size())) {
        throw std::out_of_range("Stream index out of range");
    }
    return streams_[idx];
}

void StreamPool::synchronize_all() {
    // Wait for all streams - the GPU equivalent of "are we there yet?"
    for (auto stream : streams_) {
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
}

}  // namespace rapidadb
