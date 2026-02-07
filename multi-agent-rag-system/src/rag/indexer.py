"""Vector indexer for Azure AI Search."""

import structlog
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    HnswAlgorithmConfiguration,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SimpleField,
    VectorSearch,
    VectorSearchProfile,
)

from src.config import settings
from src.rag.chunking import Chunk
from src.rag.embeddings import EmbeddingService

logger = structlog.get_logger(__name__)


class VectorIndexer:
    """Manages document indexing into Azure AI Search.

    Handles index creation, document upload with embeddings,
    and index lifecycle management.
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        index_name: str = settings.azure_search.index_name,
    ) -> None:
        self._embedding = embedding_service
        self._index_name = index_name
        credential = AzureKeyCredential(settings.azure_search.api_key)
        self._index_client = SearchIndexClient(
            endpoint=settings.azure_search.endpoint,
            credential=credential,
        )
        self._search_client = SearchClient(
            endpoint=settings.azure_search.endpoint,
            index_name=index_name,
            credential=credential,
        )

    def create_index(self, dimension: int | None = None) -> None:
        """Create or update the search index with vector search configuration."""
        dim = dimension or self._embedding.dimension

        fields = [
            SimpleField(name="id", type=SearchFieldDataType.String, key=True, filterable=True),
            SearchableField(name="content", type=SearchFieldDataType.String, analyzer_name="ja.lucene"),
            SearchableField(name="source", type=SearchFieldDataType.String, filterable=True),
            SimpleField(name="chunk_index", type=SearchFieldDataType.Int32, filterable=True),
            SimpleField(name="strategy", type=SearchFieldDataType.String, filterable=True),
            SearchField(
                name="embedding",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=dim,
                vector_search_profile_name="default-profile",
            ),
        ]

        vector_search = VectorSearch(
            algorithms=[HnswAlgorithmConfiguration(name="default-hnsw")],
            profiles=[
                VectorSearchProfile(
                    name="default-profile",
                    algorithm_configuration_name="default-hnsw",
                )
            ],
        )

        index = SearchIndex(
            name=self._index_name,
            fields=fields,
            vector_search=vector_search,
        )

        self._index_client.create_or_update_index(index)
        logger.info("index_created", index=self._index_name, dimension=dim)

    def index_chunks(self, chunks: list[Chunk], source: str) -> int:
        """Embed and upload chunks to the search index. Returns count of indexed documents."""
        texts = [c.content for c in chunks]
        embeddings = self._embedding.embed_batch(texts)

        documents = []
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            doc = {
                "id": f"{source}_{chunk.metadata.get('chunk_index', i)}",
                "content": chunk.content,
                "source": source,
                "chunk_index": chunk.metadata.get("chunk_index", i),
                "strategy": chunk.metadata.get("strategy", "unknown"),
                "embedding": emb,
            }
            documents.append(doc)

        # Upload in batches of 100
        batch_size = 100
        total = 0
        for start in range(0, len(documents), batch_size):
            batch = documents[start : start + batch_size]
            result = self._search_client.upload_documents(batch)
            succeeded = sum(1 for r in result if r.succeeded)
            total += succeeded
            logger.info("batch_indexed", start=start, succeeded=succeeded, total=len(batch))

        logger.info("indexing_complete", source=source, total_indexed=total)
        return total

    def delete_index(self) -> None:
        """Delete the search index."""
        self._index_client.delete_index(self._index_name)
        logger.info("index_deleted", index=self._index_name)
