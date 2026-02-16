"""Profile RapidaDB CUDA kernels to identify bottlenecks."""

import argparse
import time
import torch
import numpy as np
from tabulate import tabulate


def profile_distance_kernels():
    """Profile different distance computation approaches."""
    print("\n" + "="*80)
    print("DISTANCE KERNEL PROFILING")
    print("="*80 + "\n")
    
    sizes = [(10_000, 768), (50_000, 768), (100_000, 768)]
    batch_sizes = [1, 8, 32, 128]
    
    try:
        import rapidadb._C as _C
        has_rapidadb = True
    except ImportError:
        print("⚠️  RapidaDB not built. Run: pip install -e .")
        has_rapidadb = False
    
    for n, dim in sizes:
        print(f"\n📊 Dataset: {n:,} vectors × {dim}D\n")
        
        database = torch.randn(n, dim, device='cuda', dtype=torch.float32)
        
        results = []
        
        for batch_size in batch_sizes:
            queries = torch.randn(batch_size, dim, device='cuda', dtype=torch.float32)
            
            # PyTorch baseline
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(100):
                _ = torch.cdist(queries, database).pow(2)
            torch.cuda.synchronize()
            pytorch_time = (time.perf_counter() - start) / 100 * 1000
            
            if has_rapidadb:
                # RapidaDB kernel
                torch.cuda.synchronize()
                start = time.perf_counter()
                for _ in range(100):
                    _ = _C.l2_distance(queries, database)
                torch.cuda.synchronize()
                rapidadb_time = (time.perf_counter() - start) / 100 * 1000
                
                slowdown = rapidadb_time / pytorch_time
            else:
                rapidadb_time = 0
                slowdown = 0
            
            results.append({
                'Batch': batch_size,
                'PyTorch (ms)': f"{pytorch_time:.3f}",
                'RapidaDB (ms)': f"{rapidadb_time:.3f}" if has_rapidadb else "N/A",
                'Slowdown': f"{slowdown:.2f}x" if has_rapidadb else "N/A",
            })
        
        print(tabulate(results, headers='keys', tablefmt='github'))


def profile_memory_bandwidth():
    """Estimate achieved memory bandwidth."""
    print("\n" + "="*80)
    print("MEMORY BANDWIDTH ANALYSIS")
    print("="*80 + "\n")
    
    n = 100_000
    dim = 768
    batch = 32
    
    database = torch.randn(n, dim, device='cuda', dtype=torch.float32)
    queries = torch.randn(batch, dim, device='cuda', dtype=torch.float32)
    
    # Theoretical bandwidth
    props = torch.cuda.get_device_properties(0)
    theoretical_bw_gbs = props.memory_bandwidth / 1e9 if hasattr(props, 'memory_bandwidth') else 900  # Assume 900 GB/s for modern GPUs
    
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Theoretical Memory Bandwidth: ~{theoretical_bw_gbs:.0f} GB/s")
    print()
    
    # PyTorch baseline
    torch.cuda.synchronize()
    start = time.perf_counter()
    n_iters = 100
    for _ in range(n_iters):
        _ = torch.cdist(queries, database).pow(2)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    
    # Calculate data movement
    # For distance computation: read queries (batch × dim), read database (n × dim), write output (batch × n)
    bytes_read = (batch * dim + n * dim) * 4  # float32 = 4 bytes
    bytes_write = batch * n * 4
    total_bytes = bytes_read + bytes_write
    
    achieved_bw_gbs = (total_bytes * n_iters / elapsed) / 1e9
    efficiency = (achieved_bw_gbs / theoretical_bw_gbs) * 100
    
    print(f"Data per iteration:")
    print(f"  - Read queries: {batch * dim * 4 / 1e6:.2f} MB")
    print(f"  - Read database: {n * dim * 4 / 1e6:.2f} MB")
    print(f"  - Write output: {batch * n * 4 / 1e6:.2f} MB")
    print(f"  - Total: {total_bytes / 1e6:.2f} MB")
    print()
    print(f"PyTorch Performance:")
    print(f"  - Time per iteration: {elapsed / n_iters * 1000:.3f} ms")
    print(f"  - Achieved bandwidth: {achieved_bw_gbs:.1f} GB/s")
    print(f"  - Efficiency: {efficiency:.1f}%")
    
    try:
        import rapidadb._C as _C
        
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(n_iters):
            _ = _C.l2_distance(queries, database)
        torch.cuda.synchronize()
        elapsed_rapida = time.perf_counter() - start
        
        achieved_bw_rapida = (total_bytes * n_iters / elapsed_rapida) / 1e9
        efficiency_rapida = (achieved_bw_rapida / theoretical_bw_gbs) * 100
        
        print()
        print(f"RapidaDB Performance:")
        print(f"  - Time per iteration: {elapsed_rapida / n_iters * 1000:.3f} ms")
        print(f"  - Achieved bandwidth: {achieved_bw_rapida:.1f} GB/s")
        print(f"  - Efficiency: {efficiency_rapida:.1f}%")
        print()
        print(f"⚠️  RapidaDB is {elapsed_rapida / elapsed:.2f}x slower than PyTorch")
        print(f"⚠️  This suggests kernel optimization issues (memory coalescing, shared memory, etc.)")
        
    except ImportError:
        print("\n⚠️  RapidaDB not available for comparison")


def main():
    parser = argparse.ArgumentParser(description="Profile RapidaDB CUDA kernels")
    parser.add_argument("--distance", action="store_true", help="Profile distance kernels")
    parser.add_argument("--bandwidth", action="store_true", help="Profile memory bandwidth")
    parser.add_argument("--all", action="store_true", help="Run all profiling tests")
    args = parser.parse_args()
    
    if args.all or (not args.distance and not args.bandwidth):
        args.distance = True
        args.bandwidth = True
    
    if args.distance:
        profile_distance_kernels()
    
    if args.bandwidth:
        profile_memory_bandwidth()
    
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print()
    print("To deep-dive into kernel performance, use NVIDIA Nsight Compute:")
    print()
    print("  ncu --set full -o profile python -c \\")
    print("    'import torch; import rapidadb._C as C; \\")
    print("     q = torch.randn(32, 768, device=\"cuda\"); \\")
    print("     d = torch.randn(100000, 768, device=\"cuda\"); \\")
    print("     C.l2_distance(q, d)'")
    print()
    print("  ncu-ui profile.ncu-rep")
    print()
    print("Focus on:")
    print("  - Memory throughput (should be >70% of peak)")
    print("  - Warp execution efficiency")
    print("  - Shared memory bank conflicts")
    print("  - Coalesced memory access")
    print()


if __name__ == "__main__":
    main()
