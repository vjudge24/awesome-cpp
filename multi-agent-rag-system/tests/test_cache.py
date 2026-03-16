"""Tests for CAG semantic cache."""

import time

import pytest

from src.rag.cache import CacheEntry, SemanticCache, _cosine_similarity


class TestCosineSimilarity:
    def test_identical_vectors(self):
        assert _cosine_similarity([1, 0, 0], [1, 0, 0]) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        assert _cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        assert _cosine_similarity([1, 0], [-1, 0]) == pytest.approx(-1.0)

    def test_zero_vector(self):
        assert _cosine_similarity([0, 0], [1, 1]) == 0.0


class TestSemanticCache:
    def _emb(self, val: float) -> list[float]:
        """Helper to create a simple embedding."""
        return [val, val, val]

    def test_store_and_lookup_exact(self):
        cache = SemanticCache(similarity_threshold=0.9)
        cache.store("hello", self._emb(1.0), "world")
        result = cache.lookup("hello", self._emb(1.0))
        assert result is not None
        assert result.answer == "world"

    def test_lookup_miss_below_threshold(self):
        cache = SemanticCache(similarity_threshold=0.99)
        cache.store("hello", [1, 0, 0], "world")
        result = cache.lookup("different", [0, 1, 0])
        assert result is None

    def test_lookup_similar_above_threshold(self):
        cache = SemanticCache(similarity_threshold=0.9)
        cache.store("hello", [1.0, 0.1, 0.0], "world")
        result = cache.lookup("hi", [1.0, 0.15, 0.0])
        assert result is not None
        assert result.answer == "world"

    def test_hit_count_increments(self):
        cache = SemanticCache(similarity_threshold=0.9)
        cache.store("q", self._emb(1.0), "a")
        cache.lookup("q", self._emb(1.0))
        cache.lookup("q", self._emb(1.0))
        entry = cache.lookup("q", self._emb(1.0))
        assert entry is not None
        assert entry.hit_count == 3

    def test_ttl_expiry(self):
        cache = SemanticCache(similarity_threshold=0.9, ttl_seconds=0.01)
        cache.store("q", self._emb(1.0), "a")
        time.sleep(0.02)
        result = cache.lookup("q", self._emb(1.0))
        assert result is None

    def test_lru_eviction(self):
        cache = SemanticCache(similarity_threshold=0.9, max_entries=2)
        cache.store("q1", self._emb(1.0), "a1")
        cache.store("q2", self._emb(2.0), "a2")
        cache.store("q3", self._emb(3.0), "a3")
        assert cache.size == 2
        # q1 should have been evicted (LRU)
        assert cache.invalidate("q1") is False
        assert cache.invalidate("q3") is True

    def test_invalidate(self):
        cache = SemanticCache()
        cache.store("q", self._emb(1.0), "a")
        assert cache.invalidate("q") is True
        assert cache.invalidate("q") is False
        assert cache.size == 0

    def test_clear(self):
        cache = SemanticCache()
        cache.store("q1", self._emb(1.0), "a1")
        cache.store("q2", self._emb(2.0), "a2")
        cache.clear()
        assert cache.size == 0

    def test_sources_stored(self):
        cache = SemanticCache(similarity_threshold=0.9)
        cache.store("q", self._emb(1.0), "a", sources=["doc1.pdf"])
        entry = cache.lookup("q", self._emb(1.0))
        assert entry is not None
        assert entry.sources == ["doc1.pdf"]
