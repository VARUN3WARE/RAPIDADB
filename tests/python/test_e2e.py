"""End-to-end test using the high-level Python API."""

import pytest
import torch


class TestRapidaDBE2E:
    """End-to-end tests using RapidaDB Python class."""

    def test_basic_workflow(self, device):
        """Complete add → search workflow."""
        from rapidadb import RapidaDB

        db = RapidaDB(dim=128, metric="cosine")

        # Add vectors
        vectors = torch.randn(1000, 128, device=device)
        db.add(vectors)

        assert len(db) == 1000
        assert db.dim == 128
        assert db.metric == "cosine"

        # Search
        queries = torch.randn(8, 128, device=device)
        distances, indices = db.search(queries, k=10)

        assert distances.shape == (8, 10)
        assert indices.shape == (8, 10)

    def test_repr(self, device):
        """String representation is useful."""
        from rapidadb import RapidaDB

        db = RapidaDB(dim=768)
        assert "768" in repr(db)
        assert "cosine" in repr(db)

    def test_reset(self, device):
        """Reset clears the index."""
        from rapidadb import RapidaDB

        db = RapidaDB(dim=64)
        db.add(torch.randn(100, 64, device=device))
        assert len(db) == 100

        db.reset()
        assert len(db) == 0
