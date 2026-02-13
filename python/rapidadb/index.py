"""High-level Python API for RapidaDB."""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from rapidadb.config import IndexConfig, IndexType, Metric


def _metric_to_cpp(metric: Metric):
    """Convert Python Metric enum to C++ Metric enum."""
    import rapidadb._C as _C

    mapping = {
        Metric.COSINE: _C.Metric.COSINE,
        Metric.L2: _C.Metric.L2,
        Metric.DOT_PRODUCT: _C.Metric.DOT_PRODUCT,
    }
    return mapping[metric]


class RapidaDB:
    """GPU-Native Vector Database.

    A high-level interface for adding vectors and searching for
    nearest neighbors, with all operations executing on GPU.

    Example::

        db = RapidaDB(dim=768, metric='cosine')
        db.add(embeddings)  # [n, 768] CUDA tensor
        distances, indices = db.search(queries, k=10)

    Args:
        dim:        Dimensionality of vectors.
        metric:     Distance metric ('cosine', 'l2', 'dot_product').
        config:     Full IndexConfig (overrides dim/metric if provided).
    """

    def __init__(
        self,
        dim: int = 768,
        metric: str = "cosine",
        config: Optional[IndexConfig] = None,
    ):
        if config is not None:
            self._config = config
        else:
            self._config = IndexConfig(dim=dim, metric=Metric(metric))

        self._index = self._build_index()

    def _build_index(self):
        """Instantiate the appropriate C++ index backend."""
        import rapidadb._C as _C

        if self._config.index_type == IndexType.FLAT:
            return _C.FlatIndex(
                self._config.dim,
                _metric_to_cpp(self._config.metric),
            )
        else:
            # IVF and IVF-PQ will be added in later weeks
            raise NotImplementedError(
                f"Index type '{self._config.index_type.value}' not yet implemented. "
                f"Available: flat"
            )

    def add(
        self,
        vectors: torch.Tensor,
        ids: Optional[torch.Tensor] = None,
    ) -> None:
        """Add vectors to the index.

        Args:
            vectors:  [n, dim] float32 tensor. Moved to CUDA if needed.
            ids:      [n] int64 tensor of vector IDs (optional).
        """
        if not vectors.is_cuda:
            vectors = vectors.cuda()
        vectors = vectors.contiguous().float()

        if ids is not None:
            if not ids.is_cuda:
                ids = ids.cuda()
            ids = ids.contiguous().long()
            self._index.add(vectors, ids)
        else:
            self._index.add(vectors)

    def search(
        self,
        queries: torch.Tensor,
        k: int = 10,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Search for k nearest neighbors.

        Args:
            queries:  [batch_size, dim] float32 tensor. Moved to CUDA if needed.
            k:        Number of nearest neighbors to return.

        Returns:
            Tuple of (distances, indices), each [batch_size, k].
        """
        if not queries.is_cuda:
            queries = queries.cuda()
        queries = queries.contiguous().float()

        return self._index.search(queries, k)

    def reset(self) -> None:
        """Remove all vectors from the index."""
        self._index.reset()

    @property
    def size(self) -> int:
        """Number of vectors in the index."""
        return self._index.size()

    @property
    def dim(self) -> int:
        """Dimensionality of vectors."""
        return self._config.dim

    @property
    def metric(self) -> str:
        """Distance metric name."""
        return self._config.metric.value

    def __len__(self) -> int:
        return self.size

    def __repr__(self) -> str:
        return (
            f"RapidaDB(dim={self.dim}, metric='{self.metric}', "
            f"index='{self._config.index_type.value}', size={self.size})"
        )
