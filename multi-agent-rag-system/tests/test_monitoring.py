"""Tests for monitoring callbacks and metrics aggregation."""

import time

import pytest

from src.monitoring.callbacks import (
    MetricsAggregator,
    RequestMetrics,
    StepMetrics,
)


class TestStepMetrics:
    def test_latency_calculation(self):
        step = StepMetrics(name="test", start_time=1000.0, end_time=1000.5)
        assert step.latency_ms == 500.0

    def test_estimated_cost(self):
        step = StepMetrics(
            name="test",
            input_tokens=1000,
            output_tokens=500,
        )
        assert step.estimated_cost > 0

    def test_no_error_default(self):
        step = StepMetrics(name="test")
        assert step.error is None


class TestRequestMetrics:
    def test_total_tokens(self):
        metrics = RequestMetrics(
            request_id="req-1",
            steps=[
                StepMetrics(name="a", input_tokens=100, output_tokens=50),
                StepMetrics(name="b", input_tokens=200, output_tokens=100),
            ],
        )
        assert metrics.total_tokens == 450

    def test_total_cost(self):
        metrics = RequestMetrics(
            request_id="req-1",
            steps=[
                StepMetrics(name="a", input_tokens=1000, output_tokens=500),
            ],
        )
        assert metrics.total_cost > 0

    def test_to_dict(self):
        metrics = RequestMetrics(
            request_id="req-1",
            start_time=1000.0,
            end_time=1001.0,
            steps=[StepMetrics(name="step1", start_time=1000.0, end_time=1000.5)],
        )
        d = metrics.to_dict()
        assert d["request_id"] == "req-1"
        assert d["total_latency_ms"] == 1000.0
        assert len(d["steps"]) == 1


class TestMetricsAggregator:
    def test_empty_summary(self):
        agg = MetricsAggregator()
        summary = agg.get_summary()
        assert summary["total_requests"] == 0

    def test_record_and_summarize(self):
        agg = MetricsAggregator()
        for i in range(5):
            metrics = RequestMetrics(
                request_id=f"req-{i}",
                start_time=1000.0,
                end_time=1000.0 + (i + 1) * 0.1,
                steps=[
                    StepMetrics(
                        name="step",
                        start_time=1000.0,
                        end_time=1000.0 + (i + 1) * 0.1,
                        input_tokens=100,
                        output_tokens=50,
                    )
                ],
            )
            agg.record(metrics)

        summary = agg.get_summary()
        assert summary["total_requests"] == 5
        assert summary["avg_latency_ms"] > 0
        assert summary["avg_cost_usd"] > 0
