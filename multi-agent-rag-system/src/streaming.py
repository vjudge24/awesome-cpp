"""Server-Sent Events (SSE) streaming for real-time response delivery.

Implements token-by-token streaming from Azure OpenAI, wrapped as
SSE events compatible with EventSource API on the frontend.

SSE protocol:
  data: {"type": "token", "content": "Hello"}
  data: {"type": "sources", "content": ["doc.pdf"]}
  data: {"type": "metrics", "content": {...}}
  data: {"type": "done"}
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator

import structlog
from openai import AzureOpenAI

from src.config import settings

logger = structlog.get_logger(__name__)


def _sse_event(event_type: str, content: object) -> str:
    """Format a Server-Sent Event line."""
    payload = json.dumps({"type": event_type, "content": content}, ensure_ascii=False)
    return f"data: {payload}\n\n"


async def stream_rag_response(
    question: str,
    context: str,
    sources: list[str],
    conversation_history: list[dict[str, str]] | None = None,
) -> AsyncIterator[str]:
    """Stream a RAG response token-by-token as SSE events.

    Yields SSE-formatted strings:
      1. Token events as they arrive from the LLM
      2. Source citations
      3. Timing metrics
      4. Done signal

    Args:
        question: The user's question
        context: Pre-built context string from retrieval
        sources: List of source document names
        conversation_history: Previous conversation messages
    """
    start_time = time.time()
    total_tokens = 0

    client = AzureOpenAI(
        api_key=settings.azure_openai.api_key,
        api_version=settings.azure_openai.api_version,
        azure_endpoint=settings.azure_openai.endpoint,
    )

    system_prompt = (
        "You are a helpful assistant. Answer based on the provided context. "
        "Cite sources using [Source: filename] format. "
        "Match the language of the user's question."
    )

    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    if conversation_history:
        messages.extend(conversation_history[-6:])
    messages.append({
        "role": "user",
        "content": f"Context:\n{context}\n\nQuestion: {question}",
    })

    stream = client.chat.completions.create(
        model=settings.azure_openai.chat_deployment,
        messages=messages,
        temperature=0.1,
        max_tokens=1024,
        stream=True,
    )

    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            token = chunk.choices[0].delta.content
            total_tokens += 1
            yield _sse_event("token", token)

    # After streaming completes, send metadata
    elapsed_ms = (time.time() - start_time) * 1000

    yield _sse_event("sources", sources)
    yield _sse_event("metrics", {
        "latency_ms": round(elapsed_ms, 2),
        "tokens_generated": total_tokens,
    })
    yield _sse_event("done", None)

    logger.info(
        "stream_complete",
        question_len=len(question),
        tokens=total_tokens,
        latency_ms=round(elapsed_ms, 2),
    )
