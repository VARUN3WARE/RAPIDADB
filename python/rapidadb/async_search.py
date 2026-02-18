"""Async multi-stream search for high throughput."""

import torch
from typing import List, Tuple


class AsyncBatchSearcher:
    """Pipeline batch search across multiple CUDA streams.
    
    Like having multiple checkout lanes at a grocery store - much faster!
    """
    
    def __init__(self, index, num_streams: int = 4):
        self.index = index
        self.num_streams = num_streams
        self.streams = [torch.cuda.Stream() for _ in range(num_streams)]
    
    def search_batches(
        self,
        query_batches: List[torch.Tensor],
        k: int
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Search multiple batches in parallel using streams."""
        results = []
        
        # Pipeline: overlap compute and memory transfers
        for i, queries in enumerate(query_batches):
            stream_idx = i % self.num_streams
            with torch.cuda.stream(self.streams[stream_idx]):
                # Async search on this stream
                distances, indices = self.index.search(queries, k)
                results.append((distances, indices))
        
        # Synchronize all streams - the "everybody freeze!" moment
        for stream in self.streams:
            stream.synchronize()
        
        return results
    
    def __del__(self):
        # Clean up streams
        for stream in self.streams:
            stream.synchronize()


def benchmark_async_vs_sync(index, queries: torch.Tensor, k: int, batch_size: int = 32):
    """Compare async multi-stream vs synchronous search."""
    import time
    
    # Split queries into batches
    num_batches = (len(queries) + batch_size - 1) // batch_size
    batches = [queries[i*batch_size:(i+1)*batch_size] for i in range(num_batches)]
    
    # Synchronous baseline
    torch.cuda.synchronize()
    start = time.perf_counter()
    sync_results = [index.search(batch, k) for batch in batches]
    torch.cuda.synchronize()
    sync_time = time.perf_counter() - start
    
    # Async multi-stream
    searcher = AsyncBatchSearcher(index, num_streams=4)
    torch.cuda.synchronize()
    start = time.perf_counter()
    async_results = searcher.search_batches(batches, k)
    async_time = time.perf_counter() - start
    
    speedup = sync_time / async_time
    print(f"Sync time: {sync_time*1000:.2f} ms")
    print(f"Async time: {async_time*1000:.2f} ms")
    print(f"Speedup: {speedup:.2f}x")  # Hopefully > 1!
    
    return speedup
