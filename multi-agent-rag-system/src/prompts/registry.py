"""Prompt versioning and registry for reproducible prompt management.

Provides version-controlled prompt templates with metadata tracking.
Supports loading from code or JSON files, with rollback capability.

This addresses the JD requirement:
  "Implement prompt/tool/memory design and versioning to ensure reproducibility."
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class PromptVersion:
    """An immutable, versioned prompt template."""

    name: str
    version: str
    template: str
    description: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    content_hash: str = ""

    def __post_init__(self) -> None:
        if not self.content_hash:
            h = hashlib.sha256(self.template.encode()).hexdigest()[:12]
            object.__setattr__(self, "content_hash", h)

    def render(self, **kwargs: str) -> str:
        """Render the template with the given variables."""
        return self.template.format(**kwargs)

    def to_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "version": self.version,
            "template": self.template,
            "description": self.description,
            "created_at": self.created_at,
            "content_hash": self.content_hash,
        }


class PromptRegistry:
    """Central registry for versioned prompts.

    Maintains a history of all prompt versions for auditing and rollback.
    Supports persistence to/from JSON for reproducibility across deployments.

    Usage:
        registry = PromptRegistry()
        registry.register(PromptVersion(
            name="router_system",
            version="1.0.0",
            template="You are an intent classifier...",
            description="Initial router prompt"
        ))

        # Get latest version
        prompt = registry.get("router_system")

        # Get specific version
        prompt_v1 = registry.get("router_system", version="1.0.0")

        # Persist for reproducibility
        registry.save("prompts.json")
    """

    def __init__(self) -> None:
        # name -> version -> PromptVersion
        self._store: dict[str, dict[str, PromptVersion]] = {}
        # name -> latest version string
        self._latest: dict[str, str] = {}

    def register(self, prompt: PromptVersion) -> None:
        """Register a new prompt version."""
        if prompt.name not in self._store:
            self._store[prompt.name] = {}

        if prompt.version in self._store[prompt.name]:
            existing = self._store[prompt.name][prompt.version]
            if existing.content_hash != prompt.content_hash:
                raise ValueError(
                    f"Version {prompt.version} of '{prompt.name}' already exists "
                    f"with different content. Use a new version number."
                )
            return  # Idempotent re-registration

        self._store[prompt.name][prompt.version] = prompt
        self._latest[prompt.name] = prompt.version
        logger.info(
            "prompt_registered",
            name=prompt.name,
            version=prompt.version,
            hash=prompt.content_hash,
        )

    def get(self, name: str, version: str | None = None) -> PromptVersion:
        """Get a prompt by name and optional version. Returns latest if version is None."""
        if name not in self._store:
            raise KeyError(f"Prompt '{name}' not found in registry")

        v = version or self._latest[name]
        if v not in self._store[name]:
            raise KeyError(f"Version '{v}' of prompt '{name}' not found")

        return self._store[name][v]

    def list_versions(self, name: str) -> list[str]:
        """List all versions of a prompt."""
        if name not in self._store:
            return []
        return sorted(self._store[name].keys())

    def list_prompts(self) -> dict[str, str]:
        """List all prompts with their latest versions."""
        return dict(self._latest)

    def save(self, path: str | Path) -> None:
        """Persist the registry to a JSON file."""
        data: dict[str, list[dict[str, str]]] = {}
        for name, versions in self._store.items():
            data[name] = [pv.to_dict() for pv in versions.values()]

        with open(path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info("registry_saved", path=str(path), prompts=len(data))

    def load(self, path: str | Path) -> None:
        """Load prompts from a JSON file into the registry."""
        with open(path) as f:
            data = json.load(f)

        for name, versions in data.items():
            for pv_dict in versions:
                prompt = PromptVersion(
                    name=pv_dict["name"],
                    version=pv_dict["version"],
                    template=pv_dict["template"],
                    description=pv_dict.get("description", ""),
                    created_at=pv_dict.get("created_at", ""),
                    content_hash=pv_dict.get("content_hash", ""),
                )
                self.register(prompt)

        logger.info("registry_loaded", path=str(path), prompts=len(data))


# ── Default Prompts ─────────────────────────────────────────────────

_default_registry = PromptRegistry()

# Router prompt
_default_registry.register(
    PromptVersion(
        name="router_system",
        version="1.0.0",
        template=(
            "You are an intent classifier. Given a user query, classify it into "
            "one of the following categories:\n\n"
            '1. "retrieval" - The user is asking a factual question that requires '
            "looking up information from documents.\n"
            '2. "calculation" - The user needs a mathematical calculation or data analysis.\n'
            '3. "conversation" - The user is making casual conversation or asking '
            "about your capabilities.\n"
            '4. "clarification" - The query is ambiguous and needs clarification.\n\n'
            "Respond with ONLY the category name, nothing else."
        ),
        description="Intent classifier for the router agent",
    )
)

# Synthesizer prompt
_default_registry.register(
    PromptVersion(
        name="synthesizer_system",
        version="1.0.0",
        template=(
            "You are a synthesis expert. Given retrieved context and a user question, "
            "generate a comprehensive answer.\n\n"
            "Guidelines:\n"
            "- Base your answer ONLY on the provided context\n"
            "- Include inline citations using [Source: filename] format\n"
            "- If context is insufficient, clearly state what information is missing\n"
            "- Structure your answer with clear paragraphs\n"
            "- Match the language of the user's question (Japanese or English)"
        ),
        description="Synthesis prompt for answer generation",
    )
)

# Quality check prompt
_default_registry.register(
    PromptVersion(
        name="quality_check",
        version="1.0.0",
        template=(
            "You are a quality checker for RAG-generated answers.\n\n"
            "Given:\n- The original question\n- The generated answer\n- The source context\n\n"
            "Evaluate the answer on:\n"
            "1. Faithfulness: Does the answer only contain information from the context? (yes/no)\n"
            "2. Relevance: Does the answer address the question? (yes/no)\n"
            "3. Completeness: Does the answer use all relevant context? (yes/no)\n\n"
            "If any check fails, provide a brief correction suggestion.\n"
            'Respond in JSON format: {{"faithful": bool, "relevant": bool, '
            '"complete": bool, "suggestion": "..."}}'
        ),
        description="Self-evaluation prompt for quality checking",
    )
)

# Query decomposition prompt
_default_registry.register(
    PromptVersion(
        name="query_decomposition",
        version="1.0.0",
        template=(
            "You are a query decomposition expert. Given a complex question, break it "
            "down into 1-3 simpler sub-questions that, when answered together, would "
            "answer the original question.\n\n"
            "If the question is already simple enough, return it as-is.\n\n"
            "Return each sub-question on a new line, with no numbering or bullet points."
        ),
        description="Query decomposition for multi-hop retrieval",
    )
)


def get_default_registry() -> PromptRegistry:
    """Get the default prompt registry with built-in prompts."""
    return _default_registry
