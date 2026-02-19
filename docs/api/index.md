# RapidaDB API Documentation

> Fast GPU-accelerated vector database - because your vectors deserve better

## Quick Start

```python
from rapidadb import RapidaDB
import torch

# Create index
db = RapidaDB(dim=768, metric='cosine')

# Add vectors
embeddings = torch.randn(100000, 768, device='cuda')
db.add(embeddings)

# Search
queries = torch.randn(10, 768, device='cuda')
distances, indices = db.search(queries, k=10)
```

## Core API

### RapidaDB

Main vector database class.

**Constructor:**
```python
RapidaDB(dim: int, metric: str = 'cosine', device: str = 'cuda')
```

**Parameters:**
- `dim` - Vector dimensionality
- `metric` - Distance metric: 'cosine', 'l2', or 'dot'
- `device` - Compute device (default: 'cuda')

**Methods:**

#### add()
```python
db.add(vectors: torch.Tensor) -> None
```
Add vectors to the index.

#### search()
```python
db.search(queries: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]
```
Search for k nearest neighbors.

**Returns:** (distances, indices)

## Performance

6x faster than FAISS-CPU with 100% recall.

See [benchmarks](../benchmarks/README.md) for details.
