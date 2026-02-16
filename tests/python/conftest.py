"""Shared pytest fixtures for RapidaDB tests."""

import pytest
import torch


@pytest.fixture
def device():
    """Return CUDA device if available, skip test otherwise."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda")


@pytest.fixture
def random_vectors(device):
    """Factory fixture for generating random vectors."""
    def _make(n: int, dim: int = 768, normalized: bool = False):
        vecs = torch.randn(n, dim, device=device, dtype=torch.float32)
        if normalized:
            vecs = torch.nn.functional.normalize(vecs, dim=1)
        return vecs
    return _make


@pytest.fixture
def small_db(random_vectors):
    """A small database of 1000 vectors @ 128D for fast tests."""
    return random_vectors(1000, dim=128)


@pytest.fixture
def small_queries(random_vectors):
    """A small batch of 8 queries @ 128D for fast tests."""
    return random_vectors(8, dim=128)
