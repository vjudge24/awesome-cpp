"""Evaluation pipeline runner for batch evaluation of RAG system."""

import json
from dataclasses import asdict
from pathlib import Path

import structlog

from src.evaluation.metrics import EvaluationResult, RAGEvaluator

logger = structlog.get_logger(__name__)


@dataclass
class EvalDataset:
    """A dataset for evaluation."""

    questions: list[str]
    ground_truths: list[str | None]
    contexts: list[list[str]]
    answers: list[str]


from dataclasses import dataclass


@dataclass
class EvalSummary:
    """Aggregate evaluation statistics."""

    total: int
    avg_faithfulness: float
    avg_relevancy: float
    avg_context_precision: float
    avg_context_recall: float
    avg_overall: float
    results: list[EvaluationResult]


class EvaluationRunner:
    """Run batch evaluations and produce summary reports.

    Supports loading eval datasets from JSON files and producing
    structured reports for offline analysis.
    """

    def __init__(self, evaluator: RAGEvaluator | None = None) -> None:
        self._evaluator = evaluator or RAGEvaluator()

    @staticmethod
    def load_dataset(path: str | Path) -> EvalDataset:
        """Load evaluation dataset from a JSON file.

        Expected format:
        {
            "samples": [
                {
                    "question": "...",
                    "ground_truth": "...",  // optional
                    "contexts": ["...", "..."],
                    "answer": "..."
                }
            ]
        }
        """
        with open(path) as f:
            data = json.load(f)

        samples = data["samples"]
        return EvalDataset(
            questions=[s["question"] for s in samples],
            ground_truths=[s.get("ground_truth") for s in samples],
            contexts=[s["contexts"] for s in samples],
            answers=[s["answer"] for s in samples],
        )

    def run(self, dataset: EvalDataset) -> EvalSummary:
        """Evaluate all samples in the dataset and return aggregate metrics."""
        results: list[EvaluationResult] = []

        for i in range(len(dataset.questions)):
            logger.info("evaluating_sample", index=i, total=len(dataset.questions))
            result = self._evaluator.evaluate(
                question=dataset.questions[i],
                answer=dataset.answers[i],
                contexts=dataset.contexts[i],
                ground_truth=dataset.ground_truths[i],
            )
            results.append(result)

        n = len(results)
        summary = EvalSummary(
            total=n,
            avg_faithfulness=sum(r.faithfulness for r in results) / n if n else 0.0,
            avg_relevancy=sum(r.relevancy for r in results) / n if n else 0.0,
            avg_context_precision=sum(r.context_precision for r in results) / n if n else 0.0,
            avg_context_recall=sum(r.context_recall for r in results) / n if n else 0.0,
            avg_overall=sum(r.overall_score for r in results) / n if n else 0.0,
            results=results,
        )

        logger.info(
            "evaluation_run_complete",
            total=n,
            avg_overall=f"{summary.avg_overall:.3f}",
        )
        return summary

    @staticmethod
    def save_report(summary: EvalSummary, path: str | Path) -> None:
        """Save evaluation report to a JSON file."""
        report = {
            "summary": {
                "total": summary.total,
                "avg_faithfulness": summary.avg_faithfulness,
                "avg_relevancy": summary.avg_relevancy,
                "avg_context_precision": summary.avg_context_precision,
                "avg_context_recall": summary.avg_context_recall,
                "avg_overall": summary.avg_overall,
            },
            "results": [asdict(r) for r in summary.results],
        }
        with open(path, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        logger.info("report_saved", path=str(path))
