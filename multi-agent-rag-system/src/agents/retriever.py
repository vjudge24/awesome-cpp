"""Retriever agent: handles query decomposition, retrieval, and context assembly."""

import structlog
from openai import AzureOpenAI

from src.config import settings
from src.rag.retriever import HybridRetriever, RetrievedDocument
from src.rag.embeddings import EmbeddingService

logger = structlog.get_logger(__name__)

DECOMPOSE_PROMPT = """You are a query decomposition expert. Given a complex question, break it down into 1-3 simpler sub-questions that, when answered together, would answer the original question.

If the question is already simple enough, return it as-is.

Return each sub-question on a new line, with no numbering or bullet points.
"""


class RetrieverAgent:
    """Decomposes complex queries and retrieves relevant documents.

    Implements query decomposition for multi-hop questions,
    then retrieves and deduplicates results across sub-queries.
    """

    def __init__(
        self,
        hybrid_retriever: HybridRetriever | None = None,
        client: AzureOpenAI | None = None,
    ) -> None:
        self._client = client or AzureOpenAI(
            api_key=settings.azure_openai.api_key,
            api_version=settings.azure_openai.api_version,
            azure_endpoint=settings.azure_openai.endpoint,
        )
        self._deployment = settings.azure_openai.chat_deployment

        if hybrid_retriever is None:
            embedding_svc = EmbeddingService()
            self._retriever = HybridRetriever(embedding_service=embedding_svc)
        else:
            self._retriever = hybrid_retriever

    def decompose_query(self, query: str) -> list[str]:
        """Break a complex question into simpler sub-questions."""
        response = self._client.chat.completions.create(
            model=self._deployment,
            messages=[
                {"role": "system", "content": DECOMPOSE_PROMPT},
                {"role": "user", "content": query},
            ],
            temperature=0.0,
            max_tokens=256,
        )

        raw = response.choices[0].message.content or query
        sub_queries = [q.strip() for q in raw.strip().split("\n") if q.strip()]

        if not sub_queries:
            sub_queries = [query]

        logger.info("query_decomposed", original=query[:80], sub_queries=len(sub_queries))
        return sub_queries

    def retrieve(self, query: str, filter_expr: str | None = None) -> list[RetrievedDocument]:
        """Decompose query, retrieve for each sub-query, and deduplicate results."""
        sub_queries = self.decompose_query(query)

        all_docs: list[RetrievedDocument] = []
        seen_keys: set[str] = set()

        for sub_q in sub_queries:
            docs = self._retriever.retrieve(sub_q, filter_expr=filter_expr)
            for doc in docs:
                key = f"{doc.source}_{doc.chunk_index}"
                if key not in seen_keys:
                    seen_keys.add(key)
                    all_docs.append(doc)

        # Sort by score descending
        all_docs.sort(key=lambda d: d.score, reverse=True)

        logger.info(
            "retriever_agent_complete",
            sub_queries=len(sub_queries),
            total_unique_docs=len(all_docs),
        )
        return all_docs
