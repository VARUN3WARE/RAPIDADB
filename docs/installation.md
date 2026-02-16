# Installation

## Prerequisites

- NVIDIA GPU with Compute Capability ≥ 7.0 (Volta, Turing, Ampere, Hopper)
- CUDA Toolkit 12.x
- PyTorch 2.x with CUDA support
- CMake ≥ 3.24
- C++17 compiler (GCC ≥ 11 or Clang ≥ 14)

## Install from source

```bash
git clone https://github.com/your-org/rapidadb.git
cd rapidadb
pip install -e .
```

## Verify installation

```python
import torch
from rapidadb import RapidaDB

db = RapidaDB(dim=128, metric='cosine')
db.add(torch.randn(100, 128, device='cuda'))
distances, indices = db.search(torch.randn(1, 128, device='cuda'), k=5)
print(f"Top-5 distances: {distances}")
```

## Build options

### Debug build

```bash
RAPIDADB_DEBUG=1 pip install -e .
```

### Build C++ tests

```bash
mkdir build && cd build
cmake .. -DBUILD_TESTS=ON
make -j$(nproc)
ctest --verbose
```
