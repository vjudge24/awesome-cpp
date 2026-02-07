"""Tests for evaluation metrics."""

import pytest

from src.evaluation.metrics import EvaluationResult


class TestEvaluationResult:
    def test_overall_score_calculation(self):
        result = EvaluationResult(
            question="test",
            answer="test answer",
            faithfulness=1.0,
            relevancy=1.0,
            context_precision=1.0,
            context_recall=1.0,
        )
        assert result.overall_score == 1.0

    def test_overall_score_weighted(self):
        result = EvaluationResult(
            question="test",
            answer="test answer",
            faithfulness=0.8,
            relevancy=0.6,
            context_precision=0.4,
            context_recall=0.2,
        )
        expected = 0.8 * 0.3 + 0.6 * 0.3 + 0.4 * 0.2 + 0.2 * 0.2
        assert abs(result.overall_score - expected) < 1e-10

    def test_overall_score_zeros(self):
        result = EvaluationResult(
            question="test",
            answer="test",
            faithfulness=0.0,
            relevancy=0.0,
            context_precision=0.0,
            context_recall=0.0,
        )
        assert result.overall_score == 0.0
