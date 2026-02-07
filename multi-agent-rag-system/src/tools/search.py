"""Search tools for agent use — wraps retrieval as LangChain-compatible tools."""

from langchain_core.tools import tool

from src.rag.embeddings import EmbeddingService
from src.rag.retriever import HybridRetriever


@tool
def search_documents(query: str, top_k: int = 5) -> str:
    """Search the document knowledge base for relevant information.

    Args:
        query: The search query
        top_k: Number of results to return

    Returns:
        Formatted search results with source citations
    """
    embedding_svc = EmbeddingService()
    retriever = HybridRetriever(embedding_service=embedding_svc, top_k=top_k)
    docs = retriever.retrieve(query)

    if not docs:
        return "No relevant documents found."

    parts = []
    for i, doc in enumerate(docs, 1):
        parts.append(
            f"[{i}] Source: {doc.source} (score: {doc.score:.4f})\n{doc.content}"
        )
    return "\n\n".join(parts)


@tool
def search_with_filter(query: str, source_filter: str) -> str:
    """Search documents filtered by source name.

    Args:
        query: The search query
        source_filter: Filter results to this source document

    Returns:
        Formatted search results from the specified source
    """
    embedding_svc = EmbeddingService()
    retriever = HybridRetriever(embedding_service=embedding_svc)
    filter_expr = f"source eq '{source_filter}'"
    docs = retriever.retrieve(query, filter_expr=filter_expr)

    if not docs:
        return f"No results found in source '{source_filter}'."

    parts = []
    for i, doc in enumerate(docs, 1):
        parts.append(f"[{i}] Chunk {doc.chunk_index} (score: {doc.score:.4f})\n{doc.content}")
    return "\n\n".join(parts)
