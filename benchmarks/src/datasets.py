"""Dataset loaders for benchmarking."""

import os
import numpy as np
import torch
from pathlib import Path
from typing import Tuple, Optional
from urllib.request import urlretrieve
from tqdm import tqdm


class DownloadProgressBar(tqdm):
    """Progress bar for downloads."""
    
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url: str, output_path: str):
    """Download a file with progress bar."""
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        urlretrieve(url, filename=output_path, reporthook=t.update_to)


class BenchmarkDataset:
    """Base class for benchmark datasets."""
    
    def __init__(self, data_dir: str = "./data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True, parents=True)
        
    def get_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (train_vectors, test_queries, ground_truth_indices)."""
        raise NotImplementedError


class RandomDataset(BenchmarkDataset):
    """Random synthetic data for quick testing."""
    
    def __init__(
        self,
        n: int = 100_000,  # Accept 'n' for compatibility
        n_vectors: Optional[int] = None,
        n_queries: int = 1000,
        dim: int = 768,
        seed: int = 42,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.n_vectors = n_vectors if n_vectors is not None else n
        self.n_queries = n_queries
        self.dim = dim
        self.seed = seed
        
    def get_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate random vectors."""
        np.random.seed(self.seed)
        
        train = np.random.randn(self.n_vectors, self.dim).astype(np.float32)
        # Normalize for cosine similarity
        train = train / (np.linalg.norm(train, axis=1, keepdims=True) + 1e-8)
        
        queries = np.random.randn(self.n_queries, self.dim).astype(np.float32)
        queries = queries / (np.linalg.norm(queries, axis=1, keepdims=True) + 1e-8)
        
        # Compute ground truth using numpy (brute force)
        print(f"Computing ground truth for {self.n_queries} queries...")
        similarities = queries @ train.T
        ground_truth = np.argsort(-similarities, axis=1)[:, :100]  # top 100
        
        return train, queries, ground_truth


class SIFT1M(BenchmarkDataset):
    """SIFT1M ANN benchmark dataset."""
    
    BASE_URL = "ftp://ftp.irisa.fr/local/texmex/corpus/"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dataset_dir = self.data_dir / "sift1m"
        self.dataset_dir.mkdir(exist_ok=True, parents=True)
        
    def _fvecs_read(self, filename: str) -> np.ndarray:
        """Read .fvecs file format."""
        with open(filename, 'rb') as f:
            while True:
                try:
                    dim = np.fromfile(f, dtype=np.int32, count=1)[0]
                    vec = np.fromfile(f, dtype=np.float32, count=dim)
                    if len(vec) < dim:
                        break
                    yield vec
                except IndexError:
                    break
                    
    def _ivecs_read(self, filename: str) -> np.ndarray:
        """Read .ivecs file format."""
        with open(filename, 'rb') as f:
            while True:
                try:
                    dim = np.fromfile(f, dtype=np.int32, count=1)[0]
                    vec = np.fromfile(f, dtype=np.int32, count=dim)
                    if len(vec) < dim:
                        break
                    yield vec
                except IndexError:
                    break
        
    def download(self):
        """Download SIFT1M dataset."""
        files = {
            'sift_base.fvecs': 'sift_base.fvecs',
            'sift_query.fvecs': 'sift_query.fvecs',
            'sift_groundtruth.ivecs': 'sift_groundtruth.ivecs'
        }
        
        for remote_name, local_name in files.items():
            local_path = self.dataset_dir / local_name
            if local_path.exists():
                print(f"✓ {local_name} already exists")
                continue
                
            url = self.BASE_URL + remote_name
            print(f"Downloading {remote_name}...")
            try:
                download_url(url, str(local_path))
            except Exception as e:
                print(f"Failed to download {remote_name}: {e}")
                print(f"Please manually download from {url}")
                
    def get_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load SIFT1M data."""
        # Try to download if not exists
        base_file = self.dataset_dir / "sift_base.fvecs"
        if not base_file.exists():
            print("SIFT1M data not found. Attempting download...")
            self.download()
            
        # Load data
        print("Loading SIFT1M base vectors...")
        train = np.array(list(self._fvecs_read(str(self.dataset_dir / "sift_base.fvecs"))))
        
        print("Loading SIFT1M query vectors...")
        queries = np.array(list(self._fvecs_read(str(self.dataset_dir / "sift_query.fvecs"))))
        
        print("Loading SIFT1M ground truth...")
        ground_truth = np.array(list(self._ivecs_read(str(self.dataset_dir / "sift_groundtruth.ivecs"))))
        
        return train, queries, ground_truth


class SentenceEmbeddings(BenchmarkDataset):
    """Realistic sentence embeddings (768D) for RAG workloads."""
    
    def __init__(
        self,
        n_vectors: int = 100_000,
        n_queries: int = 1000,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        use_cached: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.n_vectors = n_vectors
        self.n_queries = n_queries
        self.model_name = model_name
        self.use_cached = use_cached
        
        self.cache_dir = self.data_dir / "sentence_embeddings"
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        
    def get_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate or load cached sentence embeddings."""
        cache_file = self.cache_dir / f"embeddings_{self.n_vectors}_{self.n_queries}.npz"
        
        if self.use_cached and cache_file.exists():
            print(f"Loading cached embeddings from {cache_file}")
            data = np.load(cache_file)
            return data['train'], data['queries'], data['ground_truth']
        
        # Generate random embeddings that mimic sentence-transformers distribution
        # Sentence embeddings typically have specific statistical properties
        print(f"Generating {self.n_vectors} synthetic sentence embeddings (768D)...")
        np.random.seed(42)
        
        # Generate with some structure (not pure random)
        # Use a mixture of Gaussians to simulate semantic clustering
        n_clusters = 100
        cluster_centers = np.random.randn(n_clusters, 768).astype(np.float32)
        
        train = []
        for _ in range(self.n_vectors):
            cluster_id = np.random.randint(n_clusters)
            noise = np.random.randn(768).astype(np.float32) * 0.3
            vec = cluster_centers[cluster_id] + noise
            train.append(vec)
        train = np.array(train)
        
        # Normalize
        train = train / (np.linalg.norm(train, axis=1, keepdims=True) + 1e-8)
        
        # Queries
        queries = []
        for _ in range(self.n_queries):
            cluster_id = np.random.randint(n_clusters)
            noise = np.random.randn(768).astype(np.float32) * 0.3
            vec = cluster_centers[cluster_id] + noise
            queries.append(vec)
        queries = np.array(queries)
        queries = queries / (np.linalg.norm(queries, axis=1, keepdims=True) + 1e-8)
        
        # Ground truth
        print("Computing ground truth...")
        similarities = queries @ train.T
        ground_truth = np.argsort(-similarities, axis=1)[:, :100]
        
        # Cache
        np.savez(cache_file, train=train, queries=queries, ground_truth=ground_truth)
        print(f"Cached to {cache_file}")
        
        return train, queries, ground_truth


def get_dataset(name: str, **kwargs) -> BenchmarkDataset:
    """Factory function to get a dataset by name."""
    datasets = {
        'random': RandomDataset,
        'sift1m': SIFT1M,
        'sentence': SentenceEmbeddings,
    }
    
    if name not in datasets:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(datasets.keys())}")
        
    return datasets[name](**kwargs)
