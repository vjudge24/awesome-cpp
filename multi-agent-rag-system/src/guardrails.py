"""Input / output guard rails for safety and quality."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Input Guard
# ---------------------------------------------------------------------------

_INJECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions|prompts)", re.I),
    re.compile(r"you\s+are\s+now\s+", re.I),
    re.compile(r"system\s*:\s*", re.I),
    re.compile(r"<\|im_start\|>", re.I),
    re.compile(r"\[INST\]", re.I),
]

_PII_PATTERNS: dict[str, re.Pattern[str]] = {
    "email": re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"),
    "jp_phone": re.compile(r"0\d{1,4}-?\d{1,4}-?\d{3,4}"),
    "credit_card": re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
}


@dataclass
class GuardResult:
    passed: bool
    reason: str = ""
    detected_items: list[str] = field(default_factory=list)


class InputGuard:
    """Validates user queries before they reach the LLM."""

    def __init__(self, max_query_length: int = 4000) -> None:
        self.max_query_length = max_query_length

    def check_injection(self, text: str) -> GuardResult:
        for pat in _INJECTION_PATTERNS:
            match = pat.search(text)
            if match:
                return GuardResult(
                    passed=False,
                    reason="Potential prompt injection detected",
                    detected_items=[match.group()],
                )
        return GuardResult(passed=True)

    def check_pii(self, text: str) -> GuardResult:
        found: list[str] = []
        for kind, pat in _PII_PATTERNS.items():
            matches = pat.findall(text)
            for m in matches:
                found.append(f"{kind}: {m}")
        if found:
            return GuardResult(
                passed=False, reason="PII detected in query", detected_items=found
            )
        return GuardResult(passed=True)

    def check_length(self, text: str) -> GuardResult:
        if len(text) > self.max_query_length:
            return GuardResult(
                passed=False,
                reason=f"Query exceeds max length ({len(text)} > {self.max_query_length})",
            )
        return GuardResult(passed=True)

    def validate(self, text: str) -> GuardResult:
        for check in (self.check_injection, self.check_length, self.check_pii):
            result = check(text)
            if not result.passed:
                return result
        return GuardResult(passed=True)


# ---------------------------------------------------------------------------
# Output Guard
# ---------------------------------------------------------------------------

_HALLUCINATION_INDICATORS = [
    re.compile(r"as an ai(?: language)? model", re.I),
    re.compile(r"i (?:don't|do not) have access to", re.I),
    re.compile(r"i cannot (?:browse|access|search)", re.I),
]


class OutputGuard:
    """Validates LLM output before returning to the user."""

    def check_hallucination_indicators(self, text: str) -> GuardResult:
        for pat in _HALLUCINATION_INDICATORS:
            match = pat.search(text)
            if match:
                return GuardResult(
                    passed=False,
                    reason="Potential hallucination indicator found",
                    detected_items=[match.group()],
                )
        return GuardResult(passed=True)

    def check_source_citation(
        self, text: str, expected_sources: list[str]
    ) -> GuardResult:
        if not expected_sources:
            return GuardResult(passed=True)
        cited = [s for s in expected_sources if s in text]
        if not cited:
            return GuardResult(
                passed=False,
                reason="No expected sources cited in response",
                detected_items=expected_sources,
            )
        return GuardResult(passed=True)

    def validate(
        self, text: str, expected_sources: list[str] | None = None
    ) -> GuardResult:
        hal = self.check_hallucination_indicators(text)
        if not hal.passed:
            return hal
        if expected_sources:
            src = self.check_source_citation(text, expected_sources)
            if not src.passed:
                return src
        return GuardResult(passed=True)


# ---------------------------------------------------------------------------
# Combined pipeline
# ---------------------------------------------------------------------------


class GuardRailsPipeline:
    """Convenience wrapper combining input and output guards."""

    def __init__(self, max_query_length: int = 4000) -> None:
        self.input_guard = InputGuard(max_query_length=max_query_length)
        self.output_guard = OutputGuard()

    def check_input(self, query: str) -> GuardResult:
        return self.input_guard.validate(query)

    def check_output(
        self, response: str, expected_sources: list[str] | None = None
    ) -> GuardResult:
        return self.output_guard.validate(response, expected_sources)
