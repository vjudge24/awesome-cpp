"""FastAPI application — entry point for the multi-agent RAG service."""

import time
import uuid
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.agents.graph import AgentState, build_graph
from src.config import settings
from src.memory.conversation import ConversationMemory
from src.monitoring.callbacks import MetricsAggregator, MonitoringCallback
from src.monitoring.logger import setup_logging

logger = structlog.get_logger(__name__)
metrics_aggregator = MetricsAggregator()


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    logger.info("app_starting", environment=settings.environment)
    yield
    logger.info("app_shutting_down")


app = FastAPI(
    title="Multi-Agent RAG System",
    description="Production-grade RAG with LangGraph multi-agent orchestration",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Build the LangGraph agent workflow
agent_graph = build_graph()


# ── Request/Response Models ─────────────────────────────────────────


class QueryRequest(BaseModel):
    question: str
    session_id: str | None = None
    filter_source: str | None = None
    stream: bool = False


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    session_id: str
    quality_check: dict[str, object]
    metrics: dict[str, object]


class IngestRequest(BaseModel):
    text: str
    source: str
    chunking_strategy: str = "recursive"  # "recursive" or "semantic"


class IngestResponse(BaseModel):
    source: str
    chunks_indexed: int


class HealthResponse(BaseModel):
    status: str
    environment: str


# ── Endpoints ───────────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="ok", environment=settings.environment)


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Process a user query through the multi-agent pipeline."""
    session_id = request.session_id or str(uuid.uuid4())
    request_id = str(uuid.uuid4())

    # Load conversation memory
    memory = ConversationMemory(session_id=session_id)
    history = memory.get_history()

    # Set up monitoring
    callback = MonitoringCallback(request_id=request_id)

    # Prepare initial state
    initial_state: AgentState = {
        "query": request.question,
        "intent": "",
        "messages": history,
        "documents": [],
        "answer": "",
        "sources": [],
        "quality_check": {},
        "iteration": 0,
        "error": None,
    }

    try:
        # Run the agent graph
        result = agent_graph.invoke(initial_state, config={"callbacks": [callback]})

        # Update conversation memory
        memory.add_message("user", request.question)
        memory.add_message("assistant", result["answer"])

        # Record metrics
        request_metrics = callback.get_metrics()
        metrics_aggregator.record(request_metrics)

        return QueryResponse(
            answer=result["answer"],
            sources=result.get("sources", []),
            session_id=session_id,
            quality_check=result.get("quality_check", {}),
            metrics=request_metrics.to_dict(),
        )
    except Exception as e:
        logger.error("query_failed", error=str(e), request_id=request_id)
        raise HTTPException(status_code=500, detail=f"Query processing failed: {e}")


@app.post("/ingest", response_model=IngestResponse)
async def ingest(request: IngestRequest):
    """Ingest a document into the vector store."""
    from src.rag.chunking import RecursiveChunker, SemanticChunker
    from src.rag.embeddings import EmbeddingService
    from src.rag.indexer import VectorIndexer

    embedding_svc = EmbeddingService()

    if request.chunking_strategy == "semantic":
        chunker = SemanticChunker(embedding_fn=embedding_svc.embed)
    else:
        chunker = RecursiveChunker()

    chunks = chunker.chunk(request.text, metadata={"source": request.source})
    indexer = VectorIndexer(embedding_service=embedding_svc)
    indexer.create_index()
    count = indexer.index_chunks(chunks, source=request.source)

    logger.info("document_ingested", source=request.source, chunks=count)
    return IngestResponse(source=request.source, chunks_indexed=count)


@app.get("/metrics")
async def get_metrics():
    """Return aggregated performance metrics."""
    return metrics_aggregator.get_summary()


@app.post("/query/stream")
async def query_stream(request: QueryRequest):
    """Stream a RAG response as Server-Sent Events (SSE).

    Returns token-by-token SSE events:
      data: {"type": "token", "content": "..."}
      data: {"type": "sources", "content": [...]}
      data: {"type": "metrics", "content": {...}}
      data: {"type": "done", "content": null}
    """
    from src.streaming import stream_rag_response
    from src.rag.embeddings import EmbeddingService
    from src.rag.retriever import HybridRetriever

    session_id = request.session_id or str(uuid.uuid4())
    memory = ConversationMemory(session_id=session_id)
    history = memory.get_history()

    # Retrieve context
    embedding_svc = EmbeddingService()
    retriever = HybridRetriever(embedding_service=embedding_svc)
    docs = retriever.retrieve(request.question)
    sources = list({doc.source for doc in docs})
    context = "\n---\n".join(
        f"[Source: {d.source}]\n{d.content}" for d in docs
    )

    return StreamingResponse(
        stream_rag_response(
            question=request.question,
            context=context,
            sources=sources,
            conversation_history=history,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/prompts")
async def list_prompts():
    """List all registered prompt versions."""
    from src.prompts.registry import get_default_registry

    registry = get_default_registry()
    result = {}
    for name, latest_version in registry.list_prompts().items():
        result[name] = {
            "latest_version": latest_version,
            "all_versions": registry.list_versions(name),
        }
    return result


@app.get("/prompts/{name}")
async def get_prompt(name: str, version: str | None = None):
    """Get a specific prompt by name and optional version."""
    from src.prompts.registry import get_default_registry

    registry = get_default_registry()
    try:
        prompt = registry.get(name, version=version)
        return prompt.to_dict()
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    """Clear conversation memory for a session."""
    memory = ConversationMemory(session_id=session_id)
    memory.clear()
    return {"status": "cleared", "session_id": session_id}
