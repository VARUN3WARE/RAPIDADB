"""Profile distance kernels to identify optimization opportunities."""

import torch
import rapidadb._C as _C
import subprocess
import json
import sys


def create_test_data(n_vectors, dim, batch_size):
    """Create test data for profiling."""
    queries = torch.randn(batch_size, dim, device='cuda', dtype=torch.float32)
    database = torch.randn(n_vectors, dim, device='cuda', dtype=torch.float32)
    return queries, database


def profile_kernel_with_ncu(kernel_name, queries, database):
    """Profile a kernel using Nsight Compute."""
    # Warmup
    for _ in range(5):
        if kernel_name == 'l2':
            _C.l2_distance(queries, database)
        elif kernel_name == 'cosine':
            _C.cosine_similarity(queries, database)
        elif kernel_name == 'dot':
            _C.dot_product(queries, database)
    
    torch.cuda.synchronize()
    
    # Profile with ncu
    print(f"\nProfiling {kernel_name} kernel...")
    
    # Create a simple script to run the kernel
    script = f"""
import torch
import rapidadb._C as _C

queries = torch.randn({queries.shape[0]}, {queries.shape[1]}, device='cuda', dtype=torch.float32)
database = torch.randn({database.shape[0]}, {database.shape[1]}, device='cuda', dtype=torch.float32)

# Run kernel
"""
    
    if kernel_name == 'l2':
        script += "_C.l2_distance(queries, database)\n"
    elif kernel_name == 'cosine':
        script += "_C.cosine_similarity(queries, database)\n"
    elif kernel_name == 'dot':
        script += "_C.dot_product(queries, database)\n"
    
    script += "torch.cuda.synchronize()\n"
    
    # Write script
    with open('/tmp/profile_kernel.py', 'w') as f:
        f.write(script)
    
    # Run ncu with key metrics
    cmd = [
        'ncu',
        '--metrics',
        'smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct,'
        'smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct,'
        'sm__throughput.avg.pct_of_peak_sustained_elapsed,'
        'gpu__time_duration.sum',
        '--csv',
        'python', '/tmp/profile_kernel.py'
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        return result.stdout
    except subprocess.TimeoutExpired:
        print(f"Profiling {kernel_name} timed out")
        return None


def benchmark_kernel_simple(kernel_fn, queries, database, n_iters=100):
    """Simple benchmark without ncu overhead."""
    # Warmup
    for _ in range(10):
        kernel_fn(queries, database)
    
    torch.cuda.synchronize()
    
    # Time
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(n_iters):
        kernel_fn(queries, database)
    end.record()
    
    torch.cuda.synchronize()
    elapsed_ms = start.elapsed_time(end)
    
    return elapsed_ms / n_iters


def main():
    configs = [
        {'n': 10000, 'dim': 768, 'batch': 1},
        {'n': 100000, 'dim': 768, 'batch': 1},
        {'n': 100000, 'dim': 768, 'batch': 32},
    ]
    
    print("=" * 80)
    print("Distance Kernel Profiling")
    print("=" * 80)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()
    
    for config in configs:
        n, dim, batch = config['n'], config['dim'], config['batch']
        print(f"\nConfiguration: {n:,} vectors x {dim}D, batch={batch}")
        print("-" * 80)
        
        queries, database = create_test_data(n, dim, batch)
        
        # Benchmark each kernel
        kernels = [
            ('L2 Distance', lambda q, d: _C.l2_distance(q, d)),
            ('Cosine Similarity', lambda q, d: _C.cosine_similarity(q, d)),
            ('Dot Product', lambda q, d: _C.dot_product(q, d)),
        ]
        
        for name, kernel_fn in kernels:
            avg_ms = benchmark_kernel_simple(kernel_fn, queries, database)
            
            # Calculate metrics
            total_flops = batch * n * dim * 2  # 2 ops per element (multiply + add)
            throughput_gflops = (total_flops / 1e9) / (avg_ms / 1000)
            
            # Memory bandwidth
            bytes_read = (batch * dim + n * dim) * 4  # float32
            bytes_written = batch * n * 4
            total_bytes = bytes_read + bytes_written
            bandwidth_gb_s = (total_bytes / 1e9) / (avg_ms / 1000)
            
            print(f"  {name:20s}: {avg_ms:6.3f} ms  |  {throughput_gflops:6.1f} GFLOPS  |  {bandwidth_gb_s:6.1f} GB/s")
        
        # Compare with PyTorch baseline
        print("\n  PyTorch Baseline:")
        
        # L2 distance
        avg_ms = benchmark_kernel_simple(lambda q, d: torch.cdist(q, d).pow(2), queries, database)
        total_flops = batch * n * dim * 2
        throughput_gflops = (total_flops / 1e9) / (avg_ms / 1000)
        print(f"  {'L2 (PyTorch)':20s}: {avg_ms:6.3f} ms  |  {throughput_gflops:6.1f} GFLOPS")
        
        # Dot product
        avg_ms = benchmark_kernel_simple(lambda q, d: q @ d.T, queries, database)
        print(f"  {'Dot (PyTorch)':20s}: {avg_ms:6.3f} ms  |  {throughput_gflops:6.1f} GFLOPS")


if __name__ == '__main__':
    main()
