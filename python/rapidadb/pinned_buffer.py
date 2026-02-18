"""Pinned memory buffer for faster CPU-GPU transfers.

Why use pinned memory? Because regular memory is too relaxed.
"""

import torch
import numpy as np
from typing import Optional


class PinnedBuffer:
    """Pre-allocated pinned memory buffer for batched transfers."""
    
    def __init__(self, max_batch_size: int, dim: int, dtype=torch.float32):
        """Initialize pinned memory buffer.
        
        Args:
            max_batch_size: Maximum number of vectors in a batch
            dim: Vector dimensionality
            dtype: Data type (default: float32)
        """
        self.max_batch_size = max_batch_size
        self.dim = dim
        self.dtype = dtype
        
        # Allocate pinned memory - faster than regular malloc
        self.buffer = torch.empty(
            max_batch_size, dim,
            dtype=dtype,
            pin_memory=True
        )
    
    def copy_to_device(self, data: np.ndarray, stream: Optional[torch.cuda.Stream] = None) -> torch.Tensor:
        """Copy data to GPU via pinned buffer.
        
        Args:
            data: NumPy array to transfer
            stream: CUDA stream for async copy (optional)
            
        Returns:
            GPU tensor
        """
        batch_size = data.shape[0]
        if batch_size > self.max_batch_size:
            raise ValueError(f"Batch size {batch_size} exceeds buffer size {self.max_batch_size}")
        
        # Copy to pinned buffer
        self.buffer[:batch_size] = torch.from_numpy(data)
        
        # Transfer to GPU (async if stream provided)
        if stream is not None:
            with torch.cuda.stream(stream):
                device_tensor = self.buffer[:batch_size].cuda(non_blocking=True)
        else:
            device_tensor = self.buffer[:batch_size].cuda()
        
        return device_tensor


def benchmark_pinned_memory():
    """Benchmark pinned vs regular memory transfers."""
    import time
    
    dim = 768
    batch_size = 1000
    n_iters = 100
    
    # Regular memory
    data = np.random.randn(batch_size, dim).astype(np.float32)
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_iters):
        tensor = torch.from_numpy(data).cuda()
        torch.cuda.synchronize()
    regular_time = (time.perf_counter() - start) / n_iters
    
    # Pinned memory
    buffer = PinnedBuffer(batch_size, dim)
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_iters):
        tensor = buffer.copy_to_device(data)
        torch.cuda.synchronize()
    pinned_time = (time.perf_counter() - start) / n_iters
    
    speedup = regular_time / pinned_time
    
    print("Pinned Memory Transfer Benchmark")
    print("=" * 40)
    print(f"Batch size: {batch_size} x {dim}D")
    print(f"Regular memory: {regular_time*1000:.2f} ms")
    print(f"Pinned memory:  {pinned_time*1000:.2f} ms")
    print(f"Speedup: {speedup:.2f}x")
    print("(Pinned memory is basically DMA for nerds)")
    
    return speedup


if __name__ == "__main__":
    benchmark_pinned_memory()
