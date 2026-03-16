"""A/B testing framework for prompt variants."""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass, field


@dataclass
class VariantStats:
    """Aggregated statistics for a single prompt variant."""

    name: str
    scores: list[float] = field(default_factory=list)
    latencies: list[float] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.scores)

    @property
    def mean_score(self) -> float:
        return sum(self.scores) / len(self.scores) if self.scores else 0.0

    @property
    def mean_latency(self) -> float:
        return sum(self.latencies) / len(self.latencies) if self.latencies else 0.0

    @property
    def std_score(self) -> float:
        if len(self.scores) < 2:
            return 0.0
        mean = self.mean_score
        variance = sum((s - mean) ** 2 for s in self.scores) / (len(self.scores) - 1)
        return math.sqrt(variance)


@dataclass
class PromptExperiment:
    """A/B experiment between two prompt variants.

    Uses deterministic session-based routing so the same session always
    gets the same variant.
    """

    name: str
    variant_a: str
    variant_b: str
    traffic_split: float = 0.5  # fraction routed to variant_a
    created_at: float = field(default_factory=time.time)
    stats_a: VariantStats = field(default_factory=lambda: VariantStats(name="A"))
    stats_b: VariantStats = field(default_factory=lambda: VariantStats(name="B"))

    def route(self, session_id: str) -> str:
        """Deterministically assign a session to a variant."""
        h = hashlib.md5(
            f"{self.name}:{session_id}".encode(), usedforsecurity=False
        ).hexdigest()
        bucket = int(h[:8], 16) / 0xFFFFFFFF
        return self.variant_a if bucket < self.traffic_split else self.variant_b

    def record(
        self, session_id: str, score: float, latency: float
    ) -> None:
        variant = self.route(session_id)
        stats = self.stats_a if variant == self.variant_a else self.stats_b
        stats.scores.append(score)
        stats.latencies.append(latency)

    def significance(self) -> dict:
        """Two-proportion z-test for mean score difference.

        Returns dict with z_score, p_significant (True if |z| > 1.96),
        and winner variant name.
        """
        n_a = self.stats_a.count
        n_b = self.stats_b.count
        if n_a < 2 or n_b < 2:
            return {"z_score": 0.0, "p_significant": False, "winner": None}

        mean_a = self.stats_a.mean_score
        mean_b = self.stats_b.mean_score
        std_a = self.stats_a.std_score
        std_b = self.stats_b.std_score

        se = math.sqrt(std_a**2 / n_a + std_b**2 / n_b)
        if se == 0:
            return {"z_score": 0.0, "p_significant": False, "winner": None}

        z = (mean_a - mean_b) / se
        significant = abs(z) > 1.96

        winner = None
        if significant:
            winner = "A" if mean_a > mean_b else "B"

        return {"z_score": round(z, 4), "p_significant": significant, "winner": winner}

    def summary(self) -> dict:
        return {
            "experiment": self.name,
            "variant_a": {
                "count": self.stats_a.count,
                "mean_score": round(self.stats_a.mean_score, 4),
                "mean_latency": round(self.stats_a.mean_latency, 4),
            },
            "variant_b": {
                "count": self.stats_b.count,
                "mean_score": round(self.stats_b.mean_score, 4),
                "mean_latency": round(self.stats_b.mean_latency, 4),
            },
            "significance": self.significance(),
        }
