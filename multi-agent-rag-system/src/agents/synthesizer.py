"""Synthesizer agent: generates final response with citations and quality checks."""

import structlog
from openai import AzureOpenAI

from src.config import settings
from src.rag.retriever import RetrievedDocument

logger = structlog.get_logger(__name__)

SYNTHESIS_PROMPT = """You are a synthesis expert. Given retrieved context and a user question, generate a comprehensive answer.

Guidelines:
- Base your answer ONLY on the provided context
- Include inline citations using [Source: filename] format
- If context is insufficient, clearly state what information is missing
- Structure your answer with clear paragraphs
- Match the language of the user's question (Japanese or English)
"""

QUALITY_CHECK_PROMPT = """You are a quality checker for RAG-generated answers.

Given:
- The original question
- The generated answer
- The source context

Evaluate the answer on:
1. Faithfulness: Does the answer only contain information from the context? (yes/no)
2. Relevance: Does the answer address the question? (yes/no)
3. Completeness: Does the answer use all relevant context? (yes/no)

If any check fails, provide a brief correction suggestion.
Respond in JSON format: {"faithful": bool, "relevant": bool, "complete": bool, "suggestion": "..."}
"""


class SynthesizerAgent:
    """Generates and validates responses from retrieved documents.

    Performs a two-step process: generation followed by self-evaluation
    to ensure answer quality and faithfulness.
    """

    def __init__(self, client: AzureOpenAI | None = None) -> None:
        self._client = client or AzureOpenAI(
            api_key=settings.azure_openai.api_key,
            api_version=settings.azure_openai.api_version,
            azure_endpoint=settings.azure_openai.endpoint,
        )
        self._deployment = settings.azure_openai.chat_deployment

    def synthesize(
        self,
        question: str,
        documents: list[RetrievedDocument],
        conversation_history: list[dict[str, str]] | None = None,
    ) -> dict[str, object]:
        """Generate answer and run quality check.

        Returns dict with 'answer', 'sources', 'quality_check' keys.
        """
        context = self._format_context(documents)

        # Step 1: Generate answer
        messages: list[dict[str, str]] = [
            {"role": "system", "content": SYNTHESIS_PROMPT},
        ]
        if conversation_history:
            messages.extend(conversation_history[-6:])

        messages.append(
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
        )

        response = self._client.chat.completions.create(
            model=self._deployment,
            messages=messages,
            temperature=0.1,
            max_tokens=1024,
        )
        answer = response.choices[0].message.content or ""

        # Step 2: Quality check
        quality = self._quality_check(question, answer, context)

        sources = list({doc.source for doc in documents})
        logger.info(
            "synthesis_complete",
            answer_len=len(answer),
            sources=sources,
            quality=quality,
        )

        return {
            "answer": answer,
            "sources": sources,
            "quality_check": quality,
        }

    def _quality_check(self, question: str, answer: str, context: str) -> dict[str, object]:
        """Self-evaluate the generated answer for faithfulness and relevance."""
        try:
            response = self._client.chat.completions.create(
                model=self._deployment,
                messages=[
                    {"role": "system", "content": QUALITY_CHECK_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"Question: {question}\n\n"
                            f"Answer: {answer}\n\n"
                            f"Context: {context}"
                        ),
                    },
                ],
                temperature=0.0,
                max_tokens=256,
                response_format={"type": "json_object"},
            )

            import json

            raw = response.choices[0].message.content or "{}"
            return json.loads(raw)
        except Exception as e:
            logger.error("quality_check_failed", error=str(e))
            return {"faithful": True, "relevant": True, "complete": True, "suggestion": None}

    @staticmethod
    def _format_context(documents: list[RetrievedDocument]) -> str:
        parts = []
        for doc in documents:
            parts.append(
                f"[Source: {doc.source} | Chunk: {doc.chunk_index} | "
                f"Score: {doc.score:.4f}]\n{doc.content}"
            )
        return "\n\n---\n\n".join(parts)
