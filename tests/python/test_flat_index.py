"""Tests for FlatIndex — brute-force exact search."""

import pytest
import torch


class TestFlatIndexBasic:
    """Basic functionality tests."""

    def test_create_index(self, device):
        """Index can be created with specified dimension."""
        import rapidadb._C as _C

        idx = _C.FlatIndex(128, _C.Metric.COSINE)
        assert idx.dim() == 128
        assert idx.size() == 0

    def test_add_and_size(self, device, random_vectors):
        """Adding vectors increases index size."""
        import rapidadb._C as _C

        idx = _C.FlatIndex(128, _C.Metric.COSINE)

        vecs = random_vectors(100, dim=128)
        idx.add(vecs)
        assert idx.size() == 100

        # Add more
        idx.add(random_vectors(50, dim=128))
        assert idx.size() == 150

    def test_reset(self, device, random_vectors):
        """Reset clears all vectors."""
        import rapidadb._C as _C

        idx = _C.FlatIndex(128, _C.Metric.COSINE)
        idx.add(random_vectors(100, dim=128))
        assert idx.size() == 100

        idx.reset()
        assert idx.size() == 0

    def test_repr(self, device, random_vectors):
        """String representation is informative."""
        import rapidadb._C as _C

        idx = _C.FlatIndex(768, _C.Metric.COSINE)
        idx.add(random_vectors(42, dim=768))
        s = repr(idx)
        assert "768" in s
        assert "42" in s


class TestFlatIndexSearch:
    """Search correctness tests."""

    def test_search_returns_correct_shape(self, device, random_vectors):
        """Search returns (distances, indices) with correct shapes."""
        import rapidadb._C as _C

        idx = _C.FlatIndex(128, _C.Metric.COSINE)
        idx.add(random_vectors(1000, dim=128))

        queries = random_vectors(8, dim=128)
        distances, indices = idx.search(queries, k=10)

        assert distances.shape == (8, 10)
        assert indices.shape == (8, 10)

    def test_top1_is_self(self, device):
        """When querying a vector in the database, top-1 should be itself."""
        import rapidadb._C as _C

        db = torch.randn(100, 128, device=device)
        idx = _C.FlatIndex(128, _C.Metric.COSINE)
        idx.add(db)

        # Query the first 5 vectors
        queries = db[:5]
        distances, indices = idx.search(queries, k=1)

        # Each query's nearest neighbor should be itself
        expected_ids = torch.arange(5, device=device).unsqueeze(1)
        assert torch.equal(indices, expected_ids), \
            f"Expected {expected_ids.flatten().tolist()}, got {indices.flatten().tolist()}"

    def test_cosine_100pct_recall(self, device):
        """Flat index should achieve 100% recall (it's exhaustive)."""
        import rapidadb._C as _C

        torch.manual_seed(42)
        db = torch.randn(500, 64, device=device)
        queries = torch.randn(32, 64, device=device)
        k = 10

        # RapidaDB result
        idx = _C.FlatIndex(64, _C.Metric.COSINE)
        idx.add(db)
        _, rapida_ids = idx.search(queries, k=k)

        # PyTorch reference
        q_norm = torch.nn.functional.normalize(queries, dim=1)
        d_norm = torch.nn.functional.normalize(db, dim=1)
        sims = q_norm @ d_norm.T
        _, ref_ids = sims.topk(k, dim=1)

        # Check recall
        recall = 0.0
        for i in range(32):
            v_set = set(rapida_ids[i].cpu().tolist())
            r_set = set(ref_ids[i].cpu().tolist())
            recall += len(v_set & r_set) / k
        recall /= 32

        assert recall >= 0.99, f"Recall too low: {recall:.3f}"

    def test_l2_metric(self, device, random_vectors):
        """L2 metric returns smallest distances first."""
        import rapidadb._C as _C

        db = random_vectors(200, dim=64)
        idx = _C.FlatIndex(64, _C.Metric.L2)
        idx.add(db)

        queries = random_vectors(4, dim=64)
        distances, indices = idx.search(queries, k=10)

        # Distances should be sorted ascending (smallest first)
        for i in range(4):
            dists = distances[i]
            assert (dists[:-1] <= dists[1:] + 1e-5).all(), \
                "L2 distances not sorted ascending"

    def test_dot_product_metric(self, device, random_vectors):
        """Dot product metric returns largest values first."""
        import rapidadb._C as _C

        db = random_vectors(200, dim=64)
        idx = _C.FlatIndex(64, _C.Metric.DOT_PRODUCT)
        idx.add(db)

        queries = random_vectors(4, dim=64)
        distances, indices = idx.search(queries, k=10)

        # Scores should be sorted descending (largest first)
        for i in range(4):
            scores = distances[i]
            assert (scores[:-1] >= scores[1:] - 1e-5).all(), \
                "Dot product scores not sorted descending"

    def test_custom_ids(self, device, random_vectors):
        """User-provided IDs are returned correctly."""
        import rapidadb._C as _C

        db = random_vectors(50, dim=64)
        ids = torch.arange(1000, 1050, device=device, dtype=torch.long)

        idx = _C.FlatIndex(64, _C.Metric.COSINE)
        idx.add(db, ids)

        _, result_ids = idx.search(db[:3], k=1)
        # Should return IDs from [1000, 1050), not [0, 50)
        assert (result_ids >= 1000).all() and (result_ids < 1050).all()

    def test_batch_sizes(self, device, random_vectors):
        """Works with various batch sizes."""
        import rapidadb._C as _C

        db = random_vectors(500, dim=128)
        idx = _C.FlatIndex(128, _C.Metric.COSINE)
        idx.add(db)

        for batch in [1, 4, 16, 64, 128]:
            queries = random_vectors(batch, dim=128)
            distances, indices = idx.search(queries, k=5)
            assert distances.shape == (batch, 5)
            assert indices.shape == (batch, 5)

    def test_various_k(self, device, random_vectors):
        """Works with various k values."""
        import rapidadb._C as _C

        db = random_vectors(500, dim=128)
        idx = _C.FlatIndex(128, _C.Metric.COSINE)
        idx.add(db)

        queries = random_vectors(4, dim=128)
        for k in [1, 5, 10, 50, 100, 500]:
            distances, indices = idx.search(queries, k=k)
            assert distances.shape == (4, k)
            assert indices.shape == (4, k)
