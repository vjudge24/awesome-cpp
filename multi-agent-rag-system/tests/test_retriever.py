"""Tests for the hybrid retriever's RRF merging logic."""

import pytest

from src.rag.retriever import HybridRetriever, RetrievedDocument


class TestRRFMerge:
    """Test Reciprocal Rank Fusion merging in isolation (no API calls needed)."""

    def _make_doc(
        self, source: str, chunk_index: int, score: float, method: str = "vector"
    ) -> RetrievedDocument:
        return RetrievedDocument(
            content=f"Content from {source} chunk {chunk_index}",
            score=score,
            source=source,
            chunk_index=chunk_index,
            retrieval_method=method,
        )

    def test_single_result_set(self):
        docs = [
            self._make_doc("a.pdf", 0, 0.9),
            self._make_doc("a.pdf", 1, 0.8),
            self._make_doc("b.pdf", 0, 0.7),
        ]
        merged = HybridRetriever._rrf_merge(docs)

        assert len(merged) == 3
        # First doc should have highest RRF score (rank 1)
        assert merged[0].source == "a.pdf"
        assert merged[0].chunk_index == 0

    def test_two_result_sets_boost_overlap(self):
        vector_docs = [
            self._make_doc("a.pdf", 0, 0.9, "vector"),
            self._make_doc("b.pdf", 0, 0.8, "vector"),
        ]
        keyword_docs = [
            self._make_doc("a.pdf", 0, 5.0, "keyword"),  # Same doc appears in both
            self._make_doc("c.pdf", 0, 4.0, "keyword"),
        ]
        merged = HybridRetriever._rrf_merge(vector_docs, keyword_docs)

        # a.pdf_0 appears in both sets, so it should get the highest RRF score
        assert merged[0].source == "a.pdf"
        assert merged[0].chunk_index == 0
        assert merged[0].retrieval_method == "hybrid_rrf"

    def test_deduplication(self):
        vector_docs = [self._make_doc("a.pdf", 0, 0.9)]
        keyword_docs = [self._make_doc("a.pdf", 0, 5.0)]

        merged = HybridRetriever._rrf_merge(vector_docs, keyword_docs)
        # Same document should appear only once
        assert len(merged) == 1

    def test_empty_result_sets(self):
        merged = HybridRetriever._rrf_merge([], [])
        assert merged == []

    def test_rrf_scores_are_positive(self):
        docs = [self._make_doc("a.pdf", i, 0.9 - i * 0.1) for i in range(5)]
        merged = HybridRetriever._rrf_merge(docs)

        for doc in merged:
            assert doc.score > 0
