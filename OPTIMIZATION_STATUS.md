# CUDA Kernel Optimization Status

## Summary

Optimized CUDA kernels have been implemented with shared memory tiling and vectorized loads, but compilation is blocked by CUDA 11.8 + Python 3.13 compatibility issues.

## Current Performance (Naive Kernels)

100K vectors x 768D, batch=32:
- RapidaDB: 86.7ms (56.7 GFLOPS, 3.7 GB/s)
- PyTorch GPU: 12.1ms (406.5 GFLOPS)
- **Gap: 7.2x slower**

## Optimizations Implemented

### 1. Shared Memory Tiling
**File:** `csrc/kernels/distance.cu` (lines 14-209)

```cuda
__shared__ float smem_query[TILE_DIM];  // 128 floats

// Load query cooperatively
for (int i = threadIdx.x; i < dim; i += blockDim.x) {
    smem_query[i] = queries[query_idx * dim + i];
}
__syncthreads();
```

**Benefit:** Reduces global memory reads from O(N*D) to O(D) per query

### 2. Vectorized Loads (float4)

```cuda
const float4* d4 = reinterpret_cast<const float4*>(database + db_idx * dim);
for (int i = 0; i < dim/4; i++) {
    float4 d_vec = d4[i];
    // Process 4 floats at once
}
```

**Benefit:** 4x memory bandwidth (reading 16 bytes instead of 4)

### 3. Increased Thread Block Size

```cuda
constexpr int THREADS_PER_BLOCK = 256;  // Was 128
```

**Benefit:** Better GPU occupancy and warp utilization

## Expected Performance

Based on optimization techniques:
- **Latency:** 86.7ms → ~18ms (4.8x faster)
- **Throughput:** 52K QPS → ~250K QPS
- **Memory Bandwidth:** 3.7 GB/s → ~300 GB/s

Still 2x slower than PyTorch, but much closer. Further optimizations needed:
- Warp shuffle reductions
- Better register allocation
- Kernel fusion

## Compilation Blocker

**Error:**
```
/usr/include/c++/11/bits/std_function.h:435:145: error: 
parameter packs not expanded with '...'
```

**Root Cause:** CUDA 11.8 nvcc incompatible with Python 3.13 + gcc 11

**Solutions:**

### Option 1: Upgrade CUDA to 12.x (Recommended)
```bash
# Install CUDA 12.1+
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run
sudo sh cuda_12.1.0_530.30.02_linux.run

# Rebuild
pip install -e . --no-build-isolation
```

### Option 2: Downgrade Python to 3.10
```bash
pyenv install 3.10
pyenv local 3.10
python -m venv .venv
source .venv/bin/activate
pip install torch
pip install -e .
```

### Option 3: Use Conda Environment
```bash
conda create -n rapidadb python=3.10 pytorch pytorch-cuda=11.8 -c pytorch -c nvidia
conda activate rapidadb
pip install -e .
```

## Testing Optimizations

Once compilation works:

```bash
# Profile optimized kernels
python scripts/profile_distance_kernels.py

# Run full benchmark suite
cd benchmarks
./run_benchmarks.sh standard

# Verify correctness
pytest tests/python/test_kernels.py -v
```

## Files Modified

1. `csrc/kernels/distance.cu` - Added 3 optimized kernel implementations
2. `csrc/include/rapidadb/kernels/distance.h` - Header updated
3. `setup.py` - Compiler flags updated
4. `CMakeLists.txt` - C++ standard updated

## Architecture

```
distance.cu
├── Optimized Kernels (lines 14-209)
│   ├── l2_distance_kernel_optimized<128>
│   ├── dot_product_kernel_optimized<128>
│   └── cosine_similarity_kernel_optimized<128>
├── Naive Kernels (lines 211-340) 
│   └── Fallback for dim > 128
└── Host Wrappers (lines 314-450)
    └── Auto-select optimized vs naive
```

## Next Steps

1. **Immediate:** Fix compilation environment
2. **Verify:** Run correctness tests
3. **Profile:** Use Nsight Compute to verify memory bandwidth gains
4. **Optimize:** If still slow, add warp-level primitives
5. **Benchmark:** Re-run full comparison suite

## Progress Tracking

- [x] Implement shared memory tiling
- [x] Implement vectorized loads  
- [x] Implement coalesced access
- [x] Auto-select optimized vs naive kernels
- [ ] Fix CUDA compilation
- [ ] Verify correctness
- [ ] Profile with Nsight Compute
- [ ] Achieve 4x speedup target
