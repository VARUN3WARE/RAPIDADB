"""Benchmark distance kernel performance."""

import argparse
import time

import torch


def benchmark_kernel(name, fn, queries, database, n_warmup=10, n_iters=100):
    """Benchmark a distance kernel."""
    # Warmup
    for _ in range(n_warmup):
        fn(queries, database)

    torch.cuda.synchronize()
    start = time.perf_counter()

    for _ in range(n_iters):
        fn(queries, database)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    avg_ms = elapsed / n_iters * 1000
    print(f"{name}: {avg_ms:.3f} ms (avg over {n_iters} iters)")
    return avg_ms


def main():
    parser = argparse.ArgumentParser(description="Benchmark RapidaDB distance kernels")
    parser.add_argument("--n", type=int, default=100_000, help="Number of database vectors")
    parser.add_argument("--dim", type=int, default=768, help="Vector dimensionality")
    parser.add_argument("--batch", type=int, default=1, help="Query batch size")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"Distance Kernel Benchmark")
    print(f"  Database: {args.n:,} vectors × {args.dim}D")
    print(f"  Queries:  {args.batch} × {args.dim}D")
    print(f"  Device:   {torch.cuda.get_device_name()}")
    print(f"{'='*60}\n")

    queries = torch.randn(args.batch, args.dim, device="cuda")
    database = torch.randn(args.n, args.dim, device="cuda")

    import rapidadb._C as _C

    # RapidaDB kernels
    benchmark_kernel("Cosine (RapidaDB)", _C.cosine_similarity, queries, database)
    benchmark_kernel("L2     (RapidaDB)", _C.l2_distance, queries, database)
    benchmark_kernel("Dot    (RapidaDB)", _C.dot_product, queries, database)

    # PyTorch reference
    q_norm = torch.nn.functional.normalize(queries, dim=1)
    d_norm = torch.nn.functional.normalize(database, dim=1)

    benchmark_kernel("Cosine (PyTorch) ", lambda q, d: q_norm @ d_norm.T, queries, database)
    benchmark_kernel("L2     (PyTorch) ", lambda q, d: torch.cdist(q, d).pow(2), queries, database)
    benchmark_kernel("Dot    (PyTorch) ", lambda q, d: q @ d.T, queries, database)


if __name__ == "__main__":
    main()
