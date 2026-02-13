# RapidaDB

**GPU-Native Vector Database for Production RAG & Multi-Agent Systems**

RapidaDB is a high-performance, GPU-native vector database built from scratch in C++/CUDA with PyTorch-native Python bindings. It targets sub-millisecond search latency and 100K+ QPS throughput.

## Features

- **GPU-native:** All hot-path operations execute entirely on GPU — zero host-device transfers during search
- **Multiple index types:** Flat (brute-force), IVF (partitioned), IVF-PQ (compressed)
- **Optimized CUDA kernels:** Shared memory tiling, vectorized loads, warp-level primitives
- **PyTorch-native:** Accepts and returns `torch.Tensor` — no data conversion needed
- **Mixed precision:** FP32, FP16, BF16, INT8 with adaptive precision search
- **Multi-GPU:** NCCL-based distributed search with linear throughput scaling
- **Streaming updates:** Lock-free append buffer for real-time inserts

## Quickstart

```python
import torch
from rapidadb import RapidaDB

# Create index
db = RapidaDB(dim=768, metric='cosine')

# Add vectors
embeddings = torch.randn(100_000, 768, device='cuda')
db.add(embeddings)

# Search
queries = torch.randn(32, 768, device='cuda')
distances, indices = db.search(queries, k=10)
```

## Performance

| Metric                      | RapidaDB | FAISS-GPU |
| --------------------------- | -------- | --------- |
| Latency (10M, batch=1)      | TBD      | ~1.5 ms   |
| Throughput (10M, batch=128) | TBD      | ~60K QPS  |
| Recall@10 (IVF-PQ)          | TBD      | ~95%      |

## Build

### Prerequisites

- NVIDIA GPU (Compute Capability ≥ 7.0)
- CUDA Toolkit 12.x
- PyTorch 2.x (with CUDA)
- CMake ≥ 3.24

### Install

```bash
pip install -e .
```

### Build from source

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

## Project Structure

```
RapidaDB/
├── csrc/           # C++/CUDA source code
│   ├── kernels/    # CUDA kernels (distance, top-k, PQ)
│   ├── index/      # Index implementations
│   └── bindings/   # PyTorch C++ bindings
├── python/         # Python package
├── tests/          # Unit & integration tests
├── benchmarks/     # Performance benchmarks
├── examples/       # Usage examples
└── docs/           # Documentation
```

## License

Apache License 2.0
