"""Analyze memory bandwidth and bank conflicts using PyTorch profiler."""

import torch
import rapidadb._C as _C

def profile_kernel():
    print("Memory Bandwidth & Occupancy Analysis")
    print("=" * 60)
    
    # Test configuration
    queries = torch.randn(32, 768, device='cuda')
    database = torch.randn(100_000, 768, device='cuda')
    
    # Warmup
    for _ in range(10):
        _ = _C.l2_distance(queries, database)
    
    torch.cuda.synchronize()
    
    # Profile with PyTorch profiler
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        with_stack=True
    ) as prof:
        for _ in range(100):
            _ = _C.l2_distance(queries, database)
    
    torch.cuda.synchronize()
    
    # Print results
    print("\nKernel Performance:")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    
    # Calculate theoretical bandwidth
    bytes_per_iter = (queries.numel() + database.numel()) * 4  # float32
    print(f"\nData transferred per iteration: {bytes_per_iter / 1e9:.2f} GB")
    
    # Get GPU specs
    device = torch.cuda.get_device_properties(0)
    print(f"\nGPU: {device.name}")
    print(f"Memory Clock Rate: {device.memory_clock_rate / 1e6:.2f} GHz")
    print(f"Memory Bus Width: {device.memory_bus_width} bits")
    theoretical_bw = (device.memory_clock_rate * 2 * device.memory_bus_width / 8) / 1e9
    print(f"Theoretical Bandwidth: {theoretical_bw:.2f} GB/s")

if __name__ == "__main__":
    profile_kernel()
