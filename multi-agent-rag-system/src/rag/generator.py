"""Response generation with context injection and token management."""

import tiktoken
import structlog
from openai import AzureOpenAI

from src.config import settings
from src.rag.retriever import RetrievedDocument

logger = structlog.get_logger(__name__)

SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided context.

Rules:
- Answer ONLY based on the provided context. If the context doesn't contain enough information, say so.
- Cite the source document when possible.
- Be concise and accurate.
- If the question is in Japanese, answer in Japanese. If in English, answer in English.
"""

CONTEXT_TEMPLATE = """## Retrieved Context

{context}

## Question
{question}
"""


class ResponseGenerator:
    """Generate responses using retrieved context with token budget management.

    Manages context window by tracking token counts and truncating
    context to fit within the model's limits.
    """

    def __init__(
        self,
        client: AzureOpenAI | None = None,
        deployment: str = settings.azure_openai.chat_deployment,
        max_context_tokens: int = settings.max_context_tokens,
    ) -> None:
        self._client = client or AzureOpenAI(
            api_key=settings.azure_openai.api_key,
            api_version=settings.azure_openai.api_version,
            azure_endpoint=settings.azure_openai.endpoint,
        )
        self._deployment = deployment
        self._max_context_tokens = max_context_tokens
        self._enc = tiktoken.encoding_for_model("gpt-4o")

    def generate(
        self,
        question: str,
        documents: list[RetrievedDocument],
        conversation_history: list[dict[str, str]] | None = None,
        temperature: float = 0.1,
    ) -> dict[str, str | list[str]]:
        """Generate an answer from retrieved documents.

        Returns a dict with 'answer' and 'sources' keys.
        """
        context = self._build_context(documents)
        user_content = CONTEXT_TEMPLATE.format(context=context, question=question)

        messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]

        if conversation_history:
            messages.extend(conversation_history[-6:])  # Keep last 3 turns

        messages.append({"role": "user", "content": user_content})

        total_tokens = sum(len(self._enc.encode(m["content"])) for m in messages)
        logger.info(
            "generation_request",
            question_len=len(question),
            context_docs=len(documents),
            total_input_tokens=total_tokens,
        )

        response = self._client.chat.completions.create(
            model=self._deployment,
            messages=messages,
            temperature=temperature,
            max_tokens=1024,
        )

        answer = response.choices[0].message.content or ""
        sources = list({doc.source for doc in documents})

        logger.info(
            "generation_complete",
            answer_len=len(answer),
            output_tokens=response.usage.completion_tokens if response.usage else 0,
            sources=sources,
        )

        return {"answer": answer, "sources": sources}

    def _build_context(self, documents: list[RetrievedDocument]) -> str:
        """Build context string from documents, respecting token budget."""
        parts: list[str] = []
        token_budget = self._max_context_tokens
        current_tokens = 0

        for i, doc in enumerate(documents):
            entry = f"[Source: {doc.source} | Chunk: {doc.chunk_index}]\n{doc.content}\n"
            entry_tokens = len(self._enc.encode(entry))

            if current_tokens + entry_tokens > token_budget:
                logger.info("context_truncated", included=i, total=len(documents))
                break

            parts.append(entry)
            current_tokens += entry_tokens

        return "\n---\n".join(parts)
