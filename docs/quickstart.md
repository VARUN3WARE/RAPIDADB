# Quickstart

## Create an index & search

```python
import torch
from rapidadb import RapidaDB

# Initialize a cosine similarity index
db = RapidaDB(dim=768, metric='cosine')

# Add embeddings (must be on CUDA)
embeddings = torch.randn(100_000, 768, device='cuda')
db.add(embeddings)

# Search for nearest neighbors
queries = torch.randn(32, 768, device='cuda')
distances, indices = db.search(queries, k=10)

print(f"Top-10 results shape: {indices.shape}")  # [32, 10]
```

## Supported metrics

| Metric        | Higher = better? | Use case                        |
| ------------- | ---------------- | ------------------------------- |
| `cosine`      | Yes              | Normalized embeddings (default) |
| `l2`          | No               | Euclidean distance              |
| `dot_product` | Yes              | Inner product search            |

## Index types (roadmap)

| Type     | Status    | Recall | Speed   | Memory |
| -------- | --------- | ------ | ------- | ------ |
| `flat`   | ✅ Ready  | 100%   | O(n)    | O(n·d) |
| `ivf`    | 🚧 Week 6 | ~95%   | O(n/√n) | O(n·d) |
| `ivf_pq` | 🚧 Week 8 | ~95%   | O(n/√n) | O(n·M) |
