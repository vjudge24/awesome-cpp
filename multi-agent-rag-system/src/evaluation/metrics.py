"""Evaluation metrics for RAG pipeline quality assessment.

Implements key RAG metrics:
- Faithfulness: Is the answer grounded in the retrieved context?
- Answer Relevancy: Does the answer address the question?
- Context Precision: Are the retrieved documents relevant?
- Context Recall: Was all necessary information retrieved?
"""

from dataclasses import dataclass

import structlog
from openai import AzureOpenAI

from src.config import settings

logger = structlog.get_logger(__name__)


@dataclass
class EvaluationResult:
    """Result of evaluating a single QA pair."""

    question: str
    answer: str
    faithfulness: float
    relevancy: float
    context_precision: float
    context_recall: float

    @property
    def overall_score(self) -> float:
        return (
            self.faithfulness * 0.3
            + self.relevancy * 0.3
            + self.context_precision * 0.2
            + self.context_recall * 0.2
        )


class RAGEvaluator:
    """Evaluate RAG pipeline outputs using LLM-as-judge approach.

    Each metric is computed by prompting the LLM with specific evaluation
    criteria and parsing a numeric score from the response.
    """

    def __init__(self, client: AzureOpenAI | None = None) -> None:
        self._client = client or AzureOpenAI(
            api_key=settings.azure_openai.api_key,
            api_version=settings.azure_openai.api_version,
            azure_endpoint=settings.azure_openai.endpoint,
        )
        self._deployment = settings.azure_openai.chat_deployment

    def evaluate(
        self,
        question: str,
        answer: str,
        contexts: list[str],
        ground_truth: str | None = None,
    ) -> EvaluationResult:
        """Run all evaluation metrics on a QA pair."""
        context_str = "\n---\n".join(contexts)

        faithfulness = self._score_faithfulness(answer, context_str)
        relevancy = self._score_relevancy(question, answer)
        precision = self._score_context_precision(question, contexts)
        recall = self._score_context_recall(question, context_str, ground_truth)

        result = EvaluationResult(
            question=question,
            answer=answer,
            faithfulness=faithfulness,
            relevancy=relevancy,
            context_precision=precision,
            context_recall=recall,
        )

        logger.info(
            "evaluation_complete",
            overall=f"{result.overall_score:.3f}",
            faithfulness=f"{faithfulness:.3f}",
            relevancy=f"{relevancy:.3f}",
            precision=f"{precision:.3f}",
            recall=f"{recall:.3f}",
        )
        return result

    def _llm_score(self, prompt: str) -> float:
        """Get a 0.0-1.0 score from the LLM."""
        response = self._client.chat.completions.create(
            model=self._deployment,
            messages=[
                {
                    "role": "system",
                    "content": "You are an evaluation assistant. Respond with ONLY a decimal number between 0.0 and 1.0.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=10,
        )
        raw = (response.choices[0].message.content or "0.5").strip()
        try:
            score = float(raw)
            return max(0.0, min(1.0, score))
        except ValueError:
            return 0.5

    def _score_faithfulness(self, answer: str, context: str) -> float:
        """Score: Is every claim in the answer supported by the context?"""
        prompt = (
            f"Given the context and answer below, score how faithful the answer is to the context. "
            f"1.0 means every claim is supported; 0.0 means the answer is entirely fabricated.\n\n"
            f"Context:\n{context}\n\nAnswer:\n{answer}\n\nScore:"
        )
        return self._llm_score(prompt)

    def _score_relevancy(self, question: str, answer: str) -> float:
        """Score: Does the answer actually address the question?"""
        prompt = (
            f"Given the question and answer below, score how relevant the answer is to the question. "
            f"1.0 means perfectly relevant; 0.0 means completely irrelevant.\n\n"
            f"Question:\n{question}\n\nAnswer:\n{answer}\n\nScore:"
        )
        return self._llm_score(prompt)

    def _score_context_precision(self, question: str, contexts: list[str]) -> float:
        """Score: What fraction of retrieved contexts are actually relevant to the question?"""
        if not contexts:
            return 0.0

        relevant_count = 0
        for ctx in contexts:
            prompt = (
                f"Is the following context relevant to answering the question?\n\n"
                f"Question: {question}\nContext: {ctx}\n\n"
                f"Score 1.0 if relevant, 0.0 if not:"
            )
            score = self._llm_score(prompt)
            if score >= 0.5:
                relevant_count += 1

        return relevant_count / len(contexts)

    def _score_context_recall(
        self, question: str, context: str, ground_truth: str | None
    ) -> float:
        """Score: Does the context contain all information needed to answer the question?"""
        if ground_truth is None:
            return 0.5  # Cannot evaluate without ground truth

        prompt = (
            f"Given the ground truth answer and the retrieved context, score how much of "
            f"the ground truth information is present in the context. "
            f"1.0 means all information is covered; 0.0 means none.\n\n"
            f"Question: {question}\n"
            f"Ground truth: {ground_truth}\n"
            f"Context:\n{context}\n\nScore:"
        )
        return self._llm_score(prompt)
