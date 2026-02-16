"""Comprehensive benchmark comparing RapidaDB against other vector databases."""

import argparse
import time
from typing import Dict, List, Tuple, Any
import json

import numpy as np
import torch
from tabulate import tabulate
from tqdm import tqdm

from datasets import BenchmarkDataset, get_dataset


class VectorDBBenchmark:
    """Base class for vector database benchmarks."""
    
    def __init__(self, name: str):
        self.name = name
    
    def build(self, vectors: np.ndarray, **kwargs):
        """Build the index from vectors."""
        raise NotImplementedError
    
    def search(self, queries: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Search for k nearest neighbors. Returns (distances, indices)."""
        raise NotImplementedError
    
    def cleanup(self):
        """Cleanup resources."""
        pass


class RapidaDBBenchmark(VectorDBBenchmark):
    """RapidaDB benchmark wrapper."""
    
    def __init__(self):
        super().__init__("RapidaDB")
        self.db = None
        
    def build(self, vectors: np.ndarray, metric: str = "cosine", **kwargs):
        from rapidadb import RapidaDB
        
        # Convert to torch tensor on GPU
        vectors_torch = torch.from_numpy(vectors).cuda()
        
        self.db = RapidaDB(dim=vectors.shape[1], metric=metric)
        
        build_start = time.perf_counter()
        self.db.add(vectors_torch)
        build_time = time.perf_counter() - build_start
        
        return {"build_time": build_time, "index_size_mb": 0}  # TODO: get actual size
    
    def search(self, queries: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        queries_torch = torch.from_numpy(queries).cuda()
        distances, indices = self.db.search(queries_torch, k=k)
        return distances.cpu().numpy(), indices.cpu().numpy()
    
    def cleanup(self):
        if self.db is not None:
            del self.db
            torch.cuda.empty_cache()


def measure_latency(benchmark: VectorDBBenchmark, queries: np.ndarray, k: int, 
                   n_warmup: int = 10, n_iters: int = 100) -> Dict[str, float]:
    """Measure search latency."""
    # Warmup
    for _ in range(n_warmup):
        benchmark.search(queries[:1], k)
    
    # Single query latency
    latencies = []
    for i in range(n_iters):
        start = time.perf_counter()
        benchmark.search(queries[i:i+1], k)
        latencies.append(time.perf_counter() - start)
    
    latencies = np.array(latencies) * 1000  # Convert to ms
    
    return {
        "p50_ms": np.percentile(latencies, 50),
        "p95_ms": np.percentile(latencies, 95),
        "p99_ms": np.percentile(latencies, 99),
        "mean_ms": np.mean(latencies),
    }


def measure_throughput(benchmark: VectorDBBenchmark, queries: np.ndarray, k: int,
                       batch_sizes: List[int] = [1, 8, 32, 128]) -> Dict[int, float]:
    """Measure search throughput at different batch sizes."""
    throughputs = {}
    
    for batch_size in batch_sizes:
        if batch_size > len(queries):
            continue  # Skip batch sizes larger than available queries
            
        n_batches = min(100, len(queries) // batch_size)
        if n_batches == 0:
            continue
        
        start = time.perf_counter()
        for i in range(n_batches):
            batch = queries[i*batch_size:(i+1)*batch_size]
            benchmark.search(batch, k)
        elapsed = time.perf_counter() - start
        
        qps = (n_batches * batch_size) / elapsed
        throughputs[batch_size] = qps
    
    return throughputs


def measure_recall(benchmark: VectorDBBenchmark, queries: np.ndarray, 
                   ground_truth: np.ndarray, k: int) -> float:
    """Measure recall@k against ground truth."""
    _, indices = benchmark.search(queries, k)
    
    recalls = []
    for i in range(len(queries)):
        pred = set(indices[i])
        true = set(ground_truth[i, :k])
        recall = len(pred & true) / k
        recalls.append(recall)
    
    return np.mean(recalls)


def main():
    parser = argparse.ArgumentParser(description="Benchmark RapidaDB vs competitors")
    parser.add_argument("--dataset", choices=["sift-1m", "glove-768", "random"], default="random")
    parser.add_argument("--n", type=int, default=100_000, help="Number of database vectors")
    parser.add_argument("--dim", type=int, default=768, help="Vector dimensionality")
    parser.add_argument("--n-queries", type=int, default=1000, help="Number of queries")
    parser.add_argument("--k", type=int, default=10, help="Number of neighbors to retrieve")
    parser.add_argument("--metric", choices=["cosine", "l2"], default="cosine")
    parser.add_argument("--output", type=str, default="benchmark_results.json")
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print(f"Vector Database Benchmark")
    print(f"{'='*80}")
    print(f"Dataset:        {args.dataset}")
    print(f"Database size:  {args.n:,} vectors × {args.dim}D")
    print(f"Queries:        {args.n_queries:,}")
    print(f"k:              {args.k}")
    print(f"Metric:         {args.metric}")
    print(f"{'='*80}\n")
    
    # Load dataset
    dataset = get_dataset(args.dataset, n=args.n, dim=args.dim, n_queries=args.n_queries)
    database, queries, ground_truth = dataset.get_data()
    
    results = {}
    
    # Benchmark RapidaDB
    print("\n[1/1] Benchmarking RapidaDB...")
    rapidadb = RapidaDBBenchmark()
    
    build_info = rapidadb.build(database, metric=args.metric)
    print(f"  Build time: {build_info['build_time']:.2f}s")
    
    latency = measure_latency(rapidadb, queries, args.k)
    print(f"  Latency (p50): {latency['p50_ms']:.3f}ms")
    
    throughput = measure_throughput(rapidadb, queries, args.k)
    print(f"  Throughput (batch=128): {throughput.get(128, 0):.0f} QPS")
    
    if ground_truth is not None:
        recall = measure_recall(rapidadb, queries, ground_truth, args.k)
        print(f"  Recall@{args.k}: {recall*100:.1f}%")
    else:
        recall = None
    
    results["RapidaDB"] = {
        "build_time": build_info["build_time"],
        "latency": latency,
        "throughput": throughput,
        "recall": recall,
    }
    
    rapidadb.cleanup()
    
    # Save results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to {args.output}")
    print_comparison_table(results, args.k)


def print_comparison_table(results: Dict[str, Any], k: int):
    """Print a comparison table of all benchmarks."""
    print(f"\n{'='*80}")
    print("Performance Comparison")
    print(f"{'='*80}\n")
    
    headers = ["Database", "Build (s)", "p50 (ms)", "p95 (ms)", "QPS@128", f"Recall@{k}"]
    rows = []
    
    for name, data in results.items():
        row = [
            name,
            f"{data['build_time']:.2f}",
            f"{data['latency']['p50_ms']:.3f}",
            f"{data['latency']['p95_ms']:.3f}",
            f"{data['throughput'].get(128, 0):.0f}",
            f"{data['recall']*100:.1f}%" if data['recall'] is not None else "N/A",
        ]
        rows.append(row)
    
    print(tabulate(rows, headers=headers, tablefmt="grid"))


if __name__ == "__main__":
    main()
