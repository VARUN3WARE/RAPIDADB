"""Basic search example using RapidaDB."""

import torch
from rapidadb import RapidaDB
from rapidadb.utils import cuda_timer, compute_recall


def main():
    print("RapidaDB — Basic Search Example")
    print("=" * 50)

    # Configuration
    dim = 768
    n_vectors = 100_000
    n_queries = 32
    k = 10

    # Create index
    db = RapidaDB(dim=dim, metric="cosine")
    print(f"\nCreated: {db}")

    # Generate random data
    print(f"Generating {n_vectors:,} random vectors ({dim}D)...")
    vectors = torch.randn(n_vectors, dim, device="cuda")

    # Add vectors
    with cuda_timer("Add vectors"):
        db.add(vectors)

    print(f"Index size: {len(db):,}")

    # Search
    queries = torch.randn(n_queries, dim, device="cuda")

    with cuda_timer("Search (first call)"):
        distances, indices = db.search(queries, k=k)

    # Warm up and measure
    for _ in range(10):
        db.search(queries, k=k)

    with cuda_timer(f"Search (batch={n_queries}, k={k})"):
        distances, indices = db.search(queries, k=k)

    print(f"\nResults shape: distances={distances.shape}, indices={indices.shape}")
    print(f"Top-1 distances: {distances[:5, 0].cpu().tolist()}")
    print(f"Top-1 indices:   {indices[:5, 0].cpu().tolist()}")

    # Verify recall (for flat index, should be 100%)
    # Compute ground truth with PyTorch
    q_norm = torch.nn.functional.normalize(queries, dim=1)
    d_norm = torch.nn.functional.normalize(vectors, dim=1)
    sims = q_norm @ d_norm.T
    _, gt_indices = sims.topk(k, dim=1)

    recall = compute_recall(indices, gt_indices, k=k)
    print(f"\nRecall@{k}: {recall:.4f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
