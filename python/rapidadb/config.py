"""Configuration dataclasses for RapidaDB."""

from dataclasses import dataclass, field
from enum import Enum


class Metric(str, Enum):
    """Distance metric for vector similarity search."""
    COSINE = "cosine"
    L2 = "l2"
    DOT_PRODUCT = "dot_product"


class IndexType(str, Enum):
    """Type of index to use."""
    FLAT = "flat"
    IVF = "ivf"
    IVF_PQ = "ivf_pq"


class Precision(str, Enum):
    """Numerical precision for storage and computation."""
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    INT8 = "int8"


@dataclass
class IndexConfig:
    """Configuration for RapidaDB index.

    Attributes:
        dim:            Dimensionality of vectors.
        metric:         Distance metric (cosine, l2, dot_product).
        index_type:     Index structure (flat, ivf, ivf_pq).
        precision:      Numerical precision for storage.
        nlist:          Number of IVF clusters (IVF/IVF-PQ only).
        nprobe:         Number of clusters to search (IVF/IVF-PQ only).
        pq_m:           Number of PQ subspaces (IVF-PQ only).
        pq_k:           Centroids per PQ subspace (IVF-PQ only).
        num_streams:    Number of CUDA streams for concurrent execution.
        max_batch_size: Maximum batch size for dynamic batching.
    """
    dim: int = 768
    metric: Metric = Metric.COSINE
    index_type: IndexType = IndexType.FLAT
    precision: Precision = Precision.FP32

    # IVF parameters
    nlist: int = 4096
    nprobe: int = 64

    # PQ parameters
    pq_m: int = 96
    pq_k: int = 256

    # Runtime
    num_streams: int = 4
    max_batch_size: int = 128
