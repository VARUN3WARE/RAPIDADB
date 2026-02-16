"""Full comparison benchmark across all vector databases."""

import argparse
import time
from typing import Dict, List, Any
import json

import numpy as np
from tabulate import tabulate
from tqdm import tqdm

from datasets import get_dataset
from bench_compare import measure_latency, measure_throughput, measure_recall, print_comparison_table
from bench_competitors import get_all_benchmarks


def run_full_benchmark(args):
    """Run comprehensive benchmark across all databases."""
    
    print(f"\n{'='*80}")
    print(f"Comprehensive Vector Database Benchmark")
    print(f"{'='*80}")
    print(f"Dataset:        {args.dataset}")
    print(f"Database size:  {args.n:,} vectors × {args.dim}D")
    print(f"Queries:        {args.n_queries:,}")
    print(f"k:              {args.k}")
    print(f"Metric:         {args.metric}")
    print(f"{'='*80}\n")
    
    # Load dataset
    print("📊 Loading dataset...")
    dataset = get_dataset(args.dataset, n=args.n, dim=args.dim, n_queries=args.n_queries)
    database, queries, ground_truth = dataset.get_data()
    print(f"✅ Loaded {len(database):,} database vectors, {len(queries):,} queries\n")
    
    # Get all available benchmarks
    benchmarks = get_all_benchmarks(include_gpu=True)
    
    print(f"🔍 Found {len(benchmarks)} vector databases to benchmark:")
    for i, bench in enumerate(benchmarks, 1):
        print(f"  {i}. {bench.name}")
    print()
    
    results = {}
    
    for i, benchmark in enumerate(benchmarks, 1):
        print(f"\n{'='*80}")
        print(f"[{i}/{len(benchmarks)}] Benchmarking {benchmark.name}")
        print(f"{'='*80}\n")
        
        try:
            # Build index
            print(f"  🔨 Building index...")
            build_info = benchmark.build(database, metric=args.metric)
            print(f"     ✅ Build time: {build_info['build_time']:.2f}s")
            
            # Measure latency
            print(f"  ⏱️  Measuring latency...")
            latency = measure_latency(benchmark, queries[:args.n_queries], args.k)
            print(f"     ✅ p50: {latency['p50_ms']:.3f}ms, p95: {latency['p95_ms']:.3f}ms, p99: {latency['p99_ms']:.3f}ms")
            
            # Measure throughput
            print(f"  🚀 Measuring throughput...")
            throughput = measure_throughput(benchmark, queries[:args.n_queries], args.k)
            print(f"     ✅ QPS@1: {throughput.get(1, 0):.0f}, QPS@32: {throughput.get(32, 0):.0f}, QPS@128: {throughput.get(128, 0):.0f}")
            
            # Measure recall (if ground truth available)
            recall = None
            if ground_truth is not None:
                print(f"  🎯 Measuring recall...")
                recall = measure_recall(benchmark, queries[:args.n_queries], ground_truth, args.k)
                print(f"     ✅ Recall@{args.k}: {recall*100:.1f}%")
            
            results[benchmark.name] = {
                "build_time": build_info["build_time"],
                "latency": latency,
                "throughput": throughput,
                "recall": recall,
            }
            
            # Cleanup
            benchmark.cleanup()
            
        except Exception as e:
            print(f"  ❌ Error benchmarking {benchmark.name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to {args.output}")
    
    # Print comparison
    print_comparison_table(results, args.k)
    
    # Print winner analysis
    print_winner_analysis(results)
    
    return results


def print_winner_analysis(results: Dict[str, Any]):
    """Print analysis of which database won in each category."""
    print(f"\n{'='*80}")
    print("🏆 Winner Analysis")
    print(f"{'='*80}\n")
    
    # Fastest build
    fastest_build = min(results.items(), key=lambda x: x[1]["build_time"])
    print(f"⚡ Fastest Build: {fastest_build[0]} ({fastest_build[1]['build_time']:.2f}s)")
    
    # Lowest latency (p50)
    lowest_latency = min(results.items(), key=lambda x: x[1]["latency"]["p50_ms"])
    print(f"⚡ Lowest Latency (p50): {lowest_latency[0]} ({lowest_latency[1]['latency']['p50_ms']:.3f}ms)")
    
    # Highest throughput
    highest_throughput = max(results.items(), key=lambda x: x[1]["throughput"].get(128, 0))
    print(f"⚡ Highest Throughput (batch=128): {highest_throughput[0]} ({highest_throughput[1]['throughput'].get(128, 0):.0f} QPS)")
    
    # Best recall
    if any(r["recall"] is not None for r in results.values()):
        best_recall = max(
            ((k, v) for k, v in results.items() if v["recall"] is not None),
            key=lambda x: x[1]["recall"]
        )
        print(f"⚡ Best Recall: {best_recall[0]} ({best_recall[1]['recall']*100:.1f}%)")
    
    # RapidaDB ranking
    print(f"\n📊 RapidaDB Performance:")
    rapidadb_result = results.get("RapidaDB")
    if rapidadb_result:
        # Rank in latency
        latencies = sorted([(k, v["latency"]["p50_ms"]) for k, v in results.items()], key=lambda x: x[1])
        rapidadb_latency_rank = next(i for i, (k, _) in enumerate(latencies, 1) if k == "RapidaDB")
        print(f"   Latency rank: {rapidadb_latency_rank}/{len(results)}")
        
        # Rank in throughput
        throughputs = sorted([(k, v["throughput"].get(128, 0)) for k, v in results.items()], key=lambda x: x[1], reverse=True)
        rapidadb_throughput_rank = next(i for i, (k, _) in enumerate(throughputs, 1) if k == "RapidaDB")
        print(f"   Throughput rank: {rapidadb_throughput_rank}/{len(results)}")


def main():
    parser = argparse.ArgumentParser(description="Full comparison benchmark")
    parser.add_argument("--dataset", choices=["sift-1m", "glove-768", "random"], default="random")
    parser.add_argument("--n", type=int, default=100_000, help="Number of database vectors")
    parser.add_argument("--dim", type=int, default=768, help="Vector dimensionality")
    parser.add_argument("--n-queries", type=int, default=1000, help="Number of queries")
    parser.add_argument("--k", type=int, default=10, help="Number of neighbors to retrieve")
    parser.add_argument("--metric", choices=["cosine", "l2"], default="cosine")
    parser.add_argument("--output", type=str, default="benchmark_results.json")
    args = parser.parse_args()
    
    run_full_benchmark(args)


if __name__ == "__main__":
    main()
