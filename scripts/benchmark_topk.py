"""Benchmark top-k implementations."""

import time
import torch
import rapidadb._C as _C

def benchmark_topk(name, func, distances, k, n_warmup=10, n_iters=100):
    """Benchmark a top-k function."""
    # Warmup
    for _ in range(n_warmup):
        func(distances, k, largest=False)
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    
    for _ in range(n_iters):
        func(distances, k, largest=False)
    
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    
    avg_ms = elapsed / n_iters * 1000
    print(f"{name:20s}: {avg_ms:6.3f} ms")
    return avg_ms

def main():
    batch_sizes = [1, 32, 128]
    num_vectors = 100_000
    k_values = [10, 50, 100]
    
    for batch in batch_sizes:
        for k in k_values:
            print(f"\n{'='*60}")
            print(f"Batch={batch}, Vectors={num_vectors:,}, K={k}")
            print(f"{'='*60}")
            
            distances = torch.randn(batch, num_vectors, device='cuda')
            
            thrust_time = benchmark_topk("Thrust/CUB", _C.topk_thrust, distances, k)
            warp_time = benchmark_topk("Warp Heap", _C.topk_warp_heap, distances, k)
            auto_time = benchmark_topk("Auto-select", _C.topk, distances, k)
            
            speedup = thrust_time / warp_time
            print(f"\nSpeedup (Warp/Thrust): {speedup:.2f}x")

if __name__ == "__main__":
    main()
