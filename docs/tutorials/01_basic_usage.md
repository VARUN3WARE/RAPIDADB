# Tutorial 1: Basic Usage

Learn RapidaDB in 5 minutes - no PhD required.

## Installation

```bash
pip install rapidadb
```

## Your First Search

```python
import torch
from rapidadb import RapidaDB

# Step 1: Create index
db = RapidaDB(dim=768, metric='cosine')

# Step 2: Add some vectors
# (In real life, these would be embeddings from your model)
vectors = torch.randn(10000, 768, device='cuda')
db.add(vectors)

print(f"Index has {len(db)} vectors")

# Step 3: Search
queries = torch.randn(5, 768, device='cuda')
distances, indices = db.search(queries, k=10)

print(f"Found {indices.shape} neighbors")
print(f"Top result: vector {indices[0, 0]}")
```

## Distance Metrics

```python
# Cosine similarity (default)
db_cosine = RapidaDB(dim=768, metric='cosine')

# L2 distance
db_l2 = RapidaDB(dim=768, metric='l2')

# Dot product
db_dot = RapidaDB(dim=768, metric='dot')
```

## Memory Management

```python
# Check GPU memory
import torch
print(f"GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

# For 4GB GPU: stay under 800K vectors (768D)
safe_size = 800_000
if len(vectors) < safe_size:
    db.add(vectors)
else:
    # Add in batches
    for i in range(0, len(vectors), safe_size):
        db.add(vectors[i:i+safe_size])
```

## Next Steps

- [Advanced Features](02_advanced.md)
- [Benchmarking](../benchmarks/README.md)
