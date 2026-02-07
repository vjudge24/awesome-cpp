"""Embedding service with Azure OpenAI integration and caching."""

from collections import OrderedDict

import structlog
from openai import AzureOpenAI

from src.config import settings

logger = structlog.get_logger(__name__)


class EmbeddingService:
    """Generate embeddings via Azure OpenAI with an in-memory LRU cache.

    Supports batching for throughput and caching to reduce redundant API calls.
    """

    def __init__(
        self,
        client: AzureOpenAI | None = None,
        deployment: str = settings.azure_openai.embedding_deployment,
        cache_size: int = 2048,
    ) -> None:
        self._client = client or AzureOpenAI(
            api_key=settings.azure_openai.api_key,
            api_version=settings.azure_openai.api_version,
            azure_endpoint=settings.azure_openai.endpoint,
        )
        self._deployment = deployment
        self._cache: OrderedDict[str, list[float]] = OrderedDict()
        self._cache_size = cache_size
        self._dimension: int | None = None

    @property
    def dimension(self) -> int:
        """Return the embedding dimension (lazy-detected on first call)."""
        if self._dimension is None:
            sample = self.embed("dimension probe")
            self._dimension = len(sample)
        return self._dimension

    def embed(self, text: str) -> list[float]:
        """Embed a single text string."""
        if text in self._cache:
            self._cache.move_to_end(text)
            return self._cache[text]

        result = self._embed_batch([text])[0]
        self._put_cache(text, result)
        return result

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts, using cache where possible."""
        results: list[list[float] | None] = [None] * len(texts)
        to_fetch: list[tuple[int, str]] = []

        for i, text in enumerate(texts):
            if text in self._cache:
                self._cache.move_to_end(text)
                results[i] = self._cache[text]
            else:
                to_fetch.append((i, text))

        if to_fetch:
            fetch_texts = [t for _, t in to_fetch]
            embeddings = self._embed_batch(fetch_texts)
            for (idx, text), emb in zip(to_fetch, embeddings):
                self._put_cache(text, emb)
                results[idx] = emb

        return [r for r in results if r is not None]

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Call Azure OpenAI embedding API."""
        logger.info("embedding_request", count=len(texts), deployment=self._deployment)
        response = self._client.embeddings.create(input=texts, model=self._deployment)
        return [item.embedding for item in response.data]

    def _put_cache(self, key: str, value: list[float]) -> None:
        self._cache[key] = value
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
