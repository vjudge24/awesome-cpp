"""LangChain callbacks for monitoring agent execution.

Tracks latency, token usage, costs, and error rates for each
step in the agent pipeline.
"""

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import structlog
from langchain_core.callbacks import BaseCallbackHandler

logger = structlog.get_logger(__name__)

# Azure OpenAI pricing (approximate, per 1K tokens)
PRICING = {
    "gpt-4o": {"input": 0.005, "output": 0.015},
    "text-embedding-3-large": {"input": 0.00013, "output": 0.0},
}


@dataclass
class StepMetrics:
    """Metrics for a single pipeline step."""

    name: str
    start_time: float = 0.0
    end_time: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    error: str | None = None

    @property
    def latency_ms(self) -> float:
        return (self.end_time - self.start_time) * 1000

    @property
    def estimated_cost(self) -> float:
        pricing = PRICING.get("gpt-4o", {"input": 0.005, "output": 0.015})
        return (
            self.input_tokens * pricing["input"] / 1000
            + self.output_tokens * pricing["output"] / 1000
        )


@dataclass
class RequestMetrics:
    """Aggregate metrics for a full request."""

    request_id: str
    steps: list[StepMetrics] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0

    @property
    def total_latency_ms(self) -> float:
        return (self.end_time - self.start_time) * 1000

    @property
    def total_tokens(self) -> int:
        return sum(s.input_tokens + s.output_tokens for s in self.steps)

    @property
    def total_cost(self) -> float:
        return sum(s.estimated_cost for s in self.steps)

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "total_latency_ms": round(self.total_latency_ms, 2),
            "total_tokens": self.total_tokens,
            "estimated_cost_usd": round(self.total_cost, 6),
            "steps": [
                {
                    "name": s.name,
                    "latency_ms": round(s.latency_ms, 2),
                    "tokens": s.input_tokens + s.output_tokens,
                    "error": s.error,
                }
                for s in self.steps
            ],
        }


class MonitoringCallback(BaseCallbackHandler):
    """LangChain callback handler that tracks metrics per request.

    Usage:
        cb = MonitoringCallback(request_id="req-123")
        # Pass to LangChain/LangGraph invocation
        result = chain.invoke(input, config={"callbacks": [cb]})
        metrics = cb.get_metrics()
    """

    def __init__(self, request_id: str) -> None:
        self._metrics = RequestMetrics(request_id=request_id, start_time=time.time())
        self._current_step: StepMetrics | None = None

    def on_chain_start(self, serialized: dict[str, Any], inputs: dict[str, Any], **kwargs: Any) -> None:
        name = serialized.get("name", serialized.get("id", ["unknown"])[-1])
        self._current_step = StepMetrics(name=name, start_time=time.time())

    def on_chain_end(self, outputs: Any, **kwargs: Any) -> None:
        if self._current_step:
            self._current_step.end_time = time.time()
            self._metrics.steps.append(self._current_step)
            logger.info(
                "step_complete",
                step=self._current_step.name,
                latency_ms=round(self._current_step.latency_ms, 2),
            )
            self._current_step = None

    def on_chain_error(self, error: BaseException, **kwargs: Any) -> None:
        if self._current_step:
            self._current_step.end_time = time.time()
            self._current_step.error = str(error)
            self._metrics.steps.append(self._current_step)
            logger.error("step_error", step=self._current_step.name, error=str(error))
            self._current_step = None

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        if self._current_step and hasattr(response, "llm_output"):
            usage = (response.llm_output or {}).get("token_usage", {})
            self._current_step.input_tokens += usage.get("prompt_tokens", 0)
            self._current_step.output_tokens += usage.get("completion_tokens", 0)

    def get_metrics(self) -> RequestMetrics:
        self._metrics.end_time = time.time()
        return self._metrics


class MetricsAggregator:
    """Aggregate metrics across multiple requests for monitoring dashboards."""

    def __init__(self) -> None:
        self._requests: list[RequestMetrics] = []
        self._error_counts: dict[str, int] = defaultdict(int)

    def record(self, metrics: RequestMetrics) -> None:
        self._requests.append(metrics)
        for step in metrics.steps:
            if step.error:
                self._error_counts[step.name] += 1

    def get_summary(self, last_n: int = 100) -> dict[str, Any]:
        recent = self._requests[-last_n:]
        if not recent:
            return {"total_requests": 0}

        latencies = [r.total_latency_ms for r in recent]
        costs = [r.total_cost for r in recent]

        return {
            "total_requests": len(self._requests),
            "recent_count": len(recent),
            "avg_latency_ms": round(sum(latencies) / len(latencies), 2),
            "p95_latency_ms": round(sorted(latencies)[int(len(latencies) * 0.95)], 2),
            "avg_cost_usd": round(sum(costs) / len(costs), 6),
            "total_cost_usd": round(sum(r.total_cost for r in self._requests), 4),
            "error_counts": dict(self._error_counts),
        }
