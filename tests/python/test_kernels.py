"""Tests for CUDA distance kernels — validates against PyTorch reference."""

import pytest
import torch


class TestCosineKernel:
    """Test cosine similarity CUDA kernel."""

    def test_basic_correctness(self, device, random_vectors):
        """Kernel output matches torch.nn.functional.cosine_similarity."""
        import rapidadb._C as _C

        queries = random_vectors(4, dim=128)
        database = random_vectors(100, dim=128)

        # Custom kernel
        result = _C.cosine_similarity(queries, database)

        # PyTorch reference: compute pairwise cosine similarity
        # Normalize both, then matmul
        q_norm = torch.nn.functional.normalize(queries, dim=1)
        d_norm = torch.nn.functional.normalize(database, dim=1)
        expected = q_norm @ d_norm.T

        assert result.shape == (4, 100)
        assert torch.allclose(result, expected, rtol=1e-3, atol=1e-5), \
            f"Max diff: {(result - expected).abs().max().item()}"

    def test_single_query(self, device, random_vectors):
        """Works with batch_size=1."""
        import rapidadb._C as _C

        q = random_vectors(1, dim=64)
        db = random_vectors(50, dim=64)

        result = _C.cosine_similarity(q, db)
        assert result.shape == (1, 50)

    def test_identical_vectors(self, device):
        """Identical vectors should have similarity ≈ 1.0."""
        import rapidadb._C as _C

        v = torch.randn(1, 256, device=device)
        db = v.repeat(10, 1)

        result = _C.cosine_similarity(v, db)
        assert torch.allclose(result, torch.ones_like(result), atol=1e-4)

    def test_orthogonal_vectors(self, device):
        """Orthogonal vectors should have similarity ≈ 0.0."""
        import rapidadb._C as _C

        q = torch.zeros(1, 4, device=device)
        q[0, 0] = 1.0
        db = torch.zeros(1, 4, device=device)
        db[0, 1] = 1.0

        result = _C.cosine_similarity(q, db)
        assert result.abs().item() < 1e-5

    def test_large_dim(self, device, random_vectors):
        """Works with dim=768 (production size)."""
        import rapidadb._C as _C

        q = random_vectors(2, dim=768)
        db = random_vectors(500, dim=768)

        result = _C.cosine_similarity(q, db)
        assert result.shape == (2, 500)
        # Values should be in [-1, 1]
        assert result.min() >= -1.1 and result.max() <= 1.1


class TestL2Kernel:
    """Test L2 distance CUDA kernel."""

    def test_basic_correctness(self, device, random_vectors):
        """Kernel output matches torch.cdist squared."""
        import rapidadb._C as _C

        queries = random_vectors(4, dim=128)
        database = random_vectors(100, dim=128)

        result = _C.l2_distance(queries, database)
        expected = torch.cdist(queries, database, p=2).pow(2)

        assert result.shape == (4, 100)
        assert torch.allclose(result, expected, rtol=1e-3, atol=1e-4), \
            f"Max diff: {(result - expected).abs().max().item()}"

    def test_self_distance_zero(self, device):
        """Distance of a vector to itself should be 0."""
        import rapidadb._C as _C

        v = torch.randn(1, 64, device=device)
        result = _C.l2_distance(v, v)
        assert result.item() < 1e-5

    def test_non_negative(self, device, random_vectors):
        """All L2 distances should be non-negative."""
        import rapidadb._C as _C

        q = random_vectors(8, dim=128)
        db = random_vectors(200, dim=128)

        result = _C.l2_distance(q, db)
        assert (result >= -1e-6).all()


class TestDotProductKernel:
    """Test dot product CUDA kernel."""

    def test_basic_correctness(self, device, random_vectors):
        """Kernel output matches torch matmul."""
        import rapidadb._C as _C

        queries = random_vectors(4, dim=128)
        database = random_vectors(100, dim=128)

        result = _C.dot_product(queries, database)
        expected = queries @ database.T

        assert result.shape == (4, 100)
        assert torch.allclose(result, expected, rtol=1e-3, atol=1e-4), \
            f"Max diff: {(result - expected).abs().max().item()}"


class TestKernelEdgeCases:
    """Edge cases and error handling."""

    def test_dimension_mismatch_raises(self, device):
        """Mismatched dimensions should raise an error."""
        import rapidadb._C as _C

        q = torch.randn(1, 64, device=device)
        db = torch.randn(10, 128, device=device)

        with pytest.raises(RuntimeError, match="dimension mismatch"):
            _C.cosine_similarity(q, db)

    def test_cpu_tensor_raises(self):
        """CPU tensors should raise an error."""
        import rapidadb._C as _C

        q = torch.randn(1, 64)  # CPU
        db = torch.randn(10, 64)  # CPU

        with pytest.raises(RuntimeError, match="CUDA"):
            _C.cosine_similarity(q, db)

    def test_1d_tensor_raises(self, device):
        """1D tensors should raise an error."""
        import rapidadb._C as _C

        q = torch.randn(64, device=device)
        db = torch.randn(10, 64, device=device)

        with pytest.raises(RuntimeError):
            _C.cosine_similarity(q, db)
