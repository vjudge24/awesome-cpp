"""Hybrid retriever with vector search, keyword search, and re-ranking."""

from dataclasses import dataclass

import structlog
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizableTextQuery

from src.config import settings
from src.rag.embeddings import EmbeddingService

logger = structlog.get_logger(__name__)


@dataclass
class RetrievedDocument:
    """A retrieved document with score and metadata."""

    content: str
    score: float
    source: str
    chunk_index: int
    retrieval_method: str


class HybridRetriever:
    """Hybrid retrieval combining vector search and keyword search with re-ranking.

    Implements Reciprocal Rank Fusion (RRF) to merge results from
    vector similarity search and BM25 keyword search.
    """

    RRF_K = 60  # RRF constant

    def __init__(
        self,
        embedding_service: EmbeddingService,
        index_name: str = settings.azure_search.index_name,
        top_k: int = settings.top_k,
        rerank_top_n: int = settings.rerank_top_n,
    ) -> None:
        self._embedding = embedding_service
        self._top_k = top_k
        self._rerank_top_n = rerank_top_n
        self._client = SearchClient(
            endpoint=settings.azure_search.endpoint,
            index_name=index_name,
            credential=AzureKeyCredential(settings.azure_search.api_key),
        )

    def retrieve(self, query: str, filter_expr: str | None = None) -> list[RetrievedDocument]:
        """Retrieve documents using hybrid search (vector + keyword) with RRF re-ranking."""
        vector_results = self._vector_search(query, filter_expr)
        keyword_results = self._keyword_search(query, filter_expr)

        merged = self._rrf_merge(vector_results, keyword_results)

        # Return top N after re-ranking
        top_results = merged[: self._rerank_top_n]
        logger.info(
            "retrieval_complete",
            query_len=len(query),
            vector_count=len(vector_results),
            keyword_count=len(keyword_results),
            merged_count=len(merged),
            returned=len(top_results),
        )
        return top_results

    def _vector_search(
        self, query: str, filter_expr: str | None
    ) -> list[RetrievedDocument]:
        """Perform vector similarity search."""
        query_embedding = self._embedding.embed(query)
        vector_query = VectorizableTextQuery(
            text=query, k_nearest_neighbors=self._top_k, fields="embedding"
        )

        results = self._client.search(
            search_text=None,
            vector_queries=[vector_query],
            filter=filter_expr,
            top=self._top_k,
        )

        return [
            RetrievedDocument(
                content=r["content"],
                score=r["@search.score"],
                source=r["source"],
                chunk_index=r["chunk_index"],
                retrieval_method="vector",
            )
            for r in results
        ]

    def _keyword_search(
        self, query: str, filter_expr: str | None
    ) -> list[RetrievedDocument]:
        """Perform BM25 keyword search."""
        results = self._client.search(
            search_text=query,
            filter=filter_expr,
            top=self._top_k,
            query_type="simple",
        )

        return [
            RetrievedDocument(
                content=r["content"],
                score=r["@search.score"],
                source=r["source"],
                chunk_index=r["chunk_index"],
                retrieval_method="keyword",
            )
            for r in results
        ]

    def _rrf_merge(
        self,
        *result_sets: list[RetrievedDocument],
    ) -> list[RetrievedDocument]:
        """Merge multiple result sets using Reciprocal Rank Fusion.

        RRF score = sum(1 / (k + rank_i)) across all result sets,
        where k is a constant (default 60) and rank_i is the 1-based rank
        in the i-th result set.
        """
        doc_scores: dict[str, float] = {}
        doc_map: dict[str, RetrievedDocument] = {}

        for results in result_sets:
            for rank, doc in enumerate(results, start=1):
                key = f"{doc.source}_{doc.chunk_index}"
                rrf_score = 1.0 / (self.RRF_K + rank)
                doc_scores[key] = doc_scores.get(key, 0.0) + rrf_score
                if key not in doc_map:
                    doc_map[key] = doc

        sorted_keys = sorted(doc_scores, key=lambda k: doc_scores[k], reverse=True)

        return [
            RetrievedDocument(
                content=doc_map[key].content,
                score=doc_scores[key],
                source=doc_map[key].source,
                chunk_index=doc_map[key].chunk_index,
                retrieval_method="hybrid_rrf",
            )
            for key in sorted_keys
        ]
