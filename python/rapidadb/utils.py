"""Utility functions for RapidaDB."""

import time
from contextlib import contextmanager
from typing import Optional

import torch


@contextmanager
def cuda_timer(label: str = ""):
    """Context manager to time CUDA operations accurately.

    Usage::

        with cuda_timer("search"):
            results = db.search(queries, k=10)
        # prints: search: 0.42 ms
    """
    torch.cuda.synchronize()
    start = time.perf_counter()
    yield
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000
    if label:
        print(f"{label}: {elapsed_ms:.2f} ms")


def compute_recall(
    predicted: torch.Tensor,
    ground_truth: torch.Tensor,
    k: Optional[int] = None,
) -> float:
    """Compute recall@k between predicted and ground truth indices.

    Args:
        predicted:     [batch, k_pred] tensor of predicted indices.
        ground_truth:  [batch, k_gt] tensor of ground truth indices.
        k:             If set, truncate both to top-k before comparison.

    Returns:
        Average recall@k across the batch (float in [0, 1]).
    """
    if k is not None:
        predicted = predicted[:, :k]
        ground_truth = ground_truth[:, :k]

    batch_size = predicted.size(0)
    total_recall = 0.0

    for i in range(batch_size):
        pred_set = set(predicted[i].cpu().tolist())
        gt_set = set(ground_truth[i].cpu().tolist())
        if len(gt_set) > 0:
            total_recall += len(pred_set & gt_set) / len(gt_set)

    return total_recall / batch_size


def generate_random_vectors(
    n: int,
    dim: int,
    normalized: bool = False,
    device: str = "cuda",
) -> torch.Tensor:
    """Generate random vectors for testing / benchmarking.

    Args:
        n:          Number of vectors.
        dim:        Dimensionality.
        normalized: If True, L2-normalize each vector.
        device:     Target device ('cuda' or 'cpu').

    Returns:
        [n, dim] float32 tensor.
    """
    vectors = torch.randn(n, dim, device=device, dtype=torch.float32)
    if normalized:
        vectors = torch.nn.functional.normalize(vectors, dim=1)
    return vectors
