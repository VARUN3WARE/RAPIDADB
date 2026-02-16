"""Benchmark wrappers for competitor vector databases."""

import time
from typing import Tuple, Dict
import numpy as np


class FAISSBenchmark:
    """FAISS benchmark wrapper."""
    
    def __init__(self, use_gpu: bool = True):
        self.name = "FAISS-GPU" if use_gpu else "FAISS-CPU"
        self.use_gpu = use_gpu
        self.index = None
        
    def build(self, vectors: np.ndarray, metric: str = "cosine", **kwargs):
        import faiss
        
        dim = vectors.shape[1]
        
        # Create index based on metric
        if metric == "cosine":
            # Normalize vectors for cosine similarity using inner product
            vectors_normalized = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
            index = faiss.IndexFlatIP(dim)
        elif metric == "l2":
            index = faiss.IndexFlatL2(dim)
            vectors_normalized = vectors
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        # Move to GPU if requested
        if self.use_gpu:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
        
        build_start = time.perf_counter()
        index.add(vectors_normalized.astype(np.float32))
        build_time = time.perf_counter() - build_start
        
        self.index = index
        self.metric = metric
        self.vectors = vectors_normalized
        
        return {"build_time": build_time}
    
    def search(self, queries: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        if self.metric == "cosine":
            queries_normalized = queries / np.linalg.norm(queries, axis=1, keepdims=True)
        else:
            queries_normalized = queries
            
        distances, indices = self.index.search(queries_normalized.astype(np.float32), k)
        return distances, indices
    
    def cleanup(self):
        if self.index is not None:
            del self.index


class HNSWBenchmark:
    """HNSW (hnswlib) benchmark wrapper."""
    
    def __init__(self):
        self.name = "HNSW"
        self.index = None
        
    def build(self, vectors: np.ndarray, metric: str = "cosine", M: int = 16, ef_construction: int = 200, **kwargs):
        import hnswlib
        
        dim = vectors.shape[1]
        num_elements = vectors.shape[0]
        
        # Map metric names
        space = "cosine" if metric == "cosine" else "l2"
        
        # Initialize index
        self.index = hnswlib.Index(space=space, dim=dim)
        self.index.init_index(max_elements=num_elements, ef_construction=ef_construction, M=M)
        
        build_start = time.perf_counter()
        self.index.add_items(vectors.astype(np.float32), np.arange(num_elements))
        build_time = time.perf_counter() - build_start
        
        # Set search parameters
        self.index.set_ef(50)  # ef should be >= k
        
        return {"build_time": build_time}
    
    def search(self, queries: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        # Ensure ef is large enough
        self.index.set_ef(max(50, k))
        
        indices, distances = self.index.knn_query(queries.astype(np.float32), k=k)
        return distances, indices
    
    def cleanup(self):
        if self.index is not None:
            del self.index


class AnnoyBenchmark:
    """Annoy benchmark wrapper."""
    
    def __init__(self):
        self.name = "Annoy"
        self.index = None
        
    def build(self, vectors: np.ndarray, metric: str = "cosine", n_trees: int = 10, **kwargs):
        from annoy import AnnoyIndex
        
        dim = vectors.shape[1]
        
        # Map metric names
        annoy_metric = "angular" if metric == "cosine" else "euclidean"
        
        self.index = AnnoyIndex(dim, annoy_metric)
        
        build_start = time.perf_counter()
        for i, vec in enumerate(vectors):
            self.index.add_item(i, vec)
        self.index.build(n_trees)
        build_time = time.perf_counter() - build_start
        
        return {"build_time": build_time}
    
    def search(self, queries: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        indices_list = []
        distances_list = []
        
        for query in queries:
            idx, dist = self.index.get_nns_by_vector(query, k, include_distances=True)
            indices_list.append(idx)
            distances_list.append(dist)
        
        return np.array(distances_list), np.array(indices_list)
    
    def cleanup(self):
        if self.index is not None:
            del self.index


class PyTorchBruteForceBenchmark:
    """PyTorch brute-force baseline (GPU)."""
    
    def __init__(self):
        self.name = "PyTorch-BF-GPU"
        self.vectors = None
        self.metric = None
        
    def build(self, vectors: np.ndarray, metric: str = "cosine", **kwargs):
        import torch
        
        build_start = time.perf_counter()
        self.vectors = torch.from_numpy(vectors).cuda()
        
        if metric == "cosine":
            self.vectors = torch.nn.functional.normalize(self.vectors, dim=1)
        
        self.metric = metric
        build_time = time.perf_counter() - build_start
        
        return {"build_time": build_time}
    
    def search(self, queries: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        import torch
        
        queries_torch = torch.from_numpy(queries).cuda()
        
        if self.metric == "cosine":
            queries_torch = torch.nn.functional.normalize(queries_torch, dim=1)
            # Cosine similarity
            distances = queries_torch @ self.vectors.T
            # Convert to distances (higher is better for cosine)
            topk_distances, topk_indices = torch.topk(distances, k, dim=1, largest=True)
        elif self.metric == "l2":
            # L2 distance
            distances = torch.cdist(queries_torch, self.vectors)
            topk_distances, topk_indices = torch.topk(distances, k, dim=1, largest=False)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
        
        return topk_distances.cpu().numpy(), topk_indices.cpu().numpy()
    
    def cleanup(self):
        if self.vectors is not None:
            del self.vectors
            import torch
            torch.cuda.empty_cache()


def get_all_benchmarks(include_gpu: bool = True):
    """Get all available benchmark implementations."""
    benchmarks = []
    
    # Always include RapidaDB
    from bench_compare import RapidaDBBenchmark
    benchmarks.append(RapidaDBBenchmark())
    
    # FAISS
    try:
        import faiss
        if include_gpu:
            benchmarks.append(FAISSBenchmark(use_gpu=True))
        benchmarks.append(FAISSBenchmark(use_gpu=False))
    except ImportError:
        print("⚠️  FAISS not available")
    
    # HNSW
    try:
        import hnswlib
        benchmarks.append(HNSWBenchmark())
    except ImportError:
        print("⚠️  hnswlib not available")
    
    # Annoy
    try:
        from annoy import AnnoyIndex
        benchmarks.append(AnnoyBenchmark())
    except ImportError:
        print("⚠️  Annoy not available")
    
    # PyTorch baseline
    if include_gpu:
        try:
            import torch
            if torch.cuda.is_available():
                benchmarks.append(PyTorchBruteForceBenchmark())
        except ImportError:
            print("⚠️  PyTorch not available")
    
    return benchmarks
