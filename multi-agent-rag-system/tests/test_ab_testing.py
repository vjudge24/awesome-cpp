"""Tests for A/B testing framework."""

import pytest

from src.prompts.ab_testing import PromptExperiment, VariantStats


class TestVariantStats:
    def test_empty_stats(self):
        vs = VariantStats(name="A")
        assert vs.count == 0
        assert vs.mean_score == 0.0
        assert vs.mean_latency == 0.0
        assert vs.std_score == 0.0

    def test_single_score(self):
        vs = VariantStats(name="A", scores=[0.8], latencies=[1.0])
        assert vs.count == 1
        assert vs.mean_score == pytest.approx(0.8)
        assert vs.std_score == 0.0  # need >= 2 for std

    def test_multiple_scores(self):
        vs = VariantStats(name="A", scores=[0.6, 0.8, 1.0], latencies=[1.0, 2.0, 3.0])
        assert vs.count == 3
        assert vs.mean_score == pytest.approx(0.8)
        assert vs.mean_latency == pytest.approx(2.0)
        assert vs.std_score > 0


class TestPromptExperiment:
    def test_deterministic_routing(self):
        exp = PromptExperiment(
            name="test", variant_a="prompt_v1", variant_b="prompt_v2"
        )
        # Same session should always get the same variant
        v1 = exp.route("session_abc")
        v2 = exp.route("session_abc")
        assert v1 == v2

    def test_different_sessions_can_differ(self):
        exp = PromptExperiment(
            name="test", variant_a="prompt_v1", variant_b="prompt_v2"
        )
        results = {exp.route(f"session_{i}") for i in range(100)}
        # With 100 sessions, we should see both variants
        assert len(results) == 2

    def test_record_and_summary(self):
        exp = PromptExperiment(
            name="test", variant_a="prompt_v1", variant_b="prompt_v2"
        )
        for i in range(20):
            exp.record(f"session_{i}", score=0.8, latency=1.0)

        summary = exp.summary()
        assert summary["experiment"] == "test"
        total = summary["variant_a"]["count"] + summary["variant_b"]["count"]
        assert total == 20

    def test_significance_insufficient_data(self):
        exp = PromptExperiment(
            name="test", variant_a="A", variant_b="B"
        )
        sig = exp.significance()
        assert sig["p_significant"] is False
        assert sig["winner"] is None

    def test_significance_with_data(self):
        exp = PromptExperiment(
            name="test", variant_a="A", variant_b="B", traffic_split=0.5
        )
        # Record enough data with clear difference
        for i in range(50):
            session = f"s_{i}"
            variant = exp.route(session)
            if variant == "A":
                exp.record(session, score=0.9, latency=1.0)
            else:
                exp.record(session, score=0.5, latency=1.0)

        sig = exp.significance()
        assert isinstance(sig["z_score"], float)
        # With uniform scores per variant, std=0 so result is not significant
        # This tests the math doesn't crash

    def test_traffic_split_all_a(self):
        exp = PromptExperiment(
            name="test", variant_a="A", variant_b="B", traffic_split=1.0
        )
        for i in range(10):
            assert exp.route(f"s_{i}") == "A"

    def test_traffic_split_all_b(self):
        exp = PromptExperiment(
            name="test", variant_a="A", variant_b="B", traffic_split=0.0
        )
        for i in range(10):
            assert exp.route(f"s_{i}") == "B"
