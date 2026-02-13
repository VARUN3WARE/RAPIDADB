#pragma once

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdexcept>
#include <string>

namespace rapidadb {

// ─── CUDA Error Checking ────────────────────────────────────

#define CUDA_CHECK(call)                                                       \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            throw std::runtime_error(                                           \
                std::string("CUDA error at ") + __FILE__ + ":" +                \
                std::to_string(__LINE__) + ": " + cudaGetErrorString(err));      \
        }                                                                       \
    } while (0)

#define CUDA_KERNEL_CHECK()                                                    \
    do {                                                                        \
        cudaError_t err = cudaGetLastError();                                   \
        if (err != cudaSuccess) {                                               \
            throw std::runtime_error(                                           \
                std::string("CUDA kernel error at ") + __FILE__ + ":" +         \
                std::to_string(__LINE__) + ": " + cudaGetErrorString(err));      \
        }                                                                       \
    } while (0)

// ─── Utility Macros ─────────────────────────────────────────

// Integer ceiling division
#define CDIV(a, b) (((a) + (b) - 1) / (b))

// Warp size
constexpr int WARP_SIZE = 32;

// Default block size for most kernels
constexpr int DEFAULT_BLOCK_SIZE = 256;

// ─── Timer Utility ──────────────────────────────────────────

class CUDATimer {
public:
    CUDATimer() {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }

    ~CUDATimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }

    void start(cudaStream_t stream = 0) {
        CUDA_CHECK(cudaEventRecord(start_, stream));
    }

    void stop(cudaStream_t stream = 0) {
        CUDA_CHECK(cudaEventRecord(stop_, stream));
    }

    // Returns elapsed time in milliseconds
    float elapsed_ms() {
        CUDA_CHECK(cudaEventSynchronize(stop_));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
        return ms;
    }

private:
    cudaEvent_t start_;
    cudaEvent_t stop_;
};

// ─── Device Properties Helper ───────────────────────────────

inline int get_sm_count(int device = 0) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    return prop.multiProcessorCount;
}

inline size_t get_shared_mem_per_block(int device = 0) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    return prop.sharedMemPerBlock;
}

}  // namespace rapidadb
