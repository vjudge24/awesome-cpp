"""Document chunking strategies for RAG pipeline.

Supports multiple strategies: recursive character splitting and semantic chunking.
Each strategy produces chunks with metadata for traceability.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import tiktoken

from src.config import settings


@dataclass
class Chunk:
    """A document chunk with content and metadata."""

    content: str
    metadata: dict[str, str | int | float] = field(default_factory=dict)
    token_count: int = 0

    def __post_init__(self) -> None:
        if self.token_count == 0:
            enc = tiktoken.encoding_for_model("gpt-4o")
            self.token_count = len(enc.encode(self.content))


class ChunkingStrategy(ABC):
    """Abstract base for chunking strategies."""

    @abstractmethod
    def chunk(self, text: str, metadata: dict[str, str] | None = None) -> list[Chunk]:
        """Split text into chunks."""
        ...


class RecursiveChunker(ChunkingStrategy):
    """Recursive character text splitting with configurable separators.

    Splits on paragraph boundaries first, then sentences, then words,
    ensuring each chunk stays within the token limit.
    """

    SEPARATORS = ["\n\n", "\n", "。", ". ", "、", ", ", " ", ""]

    def __init__(
        self,
        chunk_size: int = settings.chunk_size,
        chunk_overlap: int = settings.chunk_overlap,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._enc = tiktoken.encoding_for_model("gpt-4o")

    def _token_len(self, text: str) -> int:
        return len(self._enc.encode(text))

    def _split_text(self, text: str, separators: list[str]) -> list[str]:
        if not separators:
            return [text]

        sep = separators[0]
        remaining_seps = separators[1:]

        if sep == "":
            # Character-level split as last resort
            pieces: list[str] = []
            current = ""
            for char in text:
                if self._token_len(current + char) > self.chunk_size:
                    if current:
                        pieces.append(current)
                    current = char
                else:
                    current += char
            if current:
                pieces.append(current)
            return pieces

        parts = text.split(sep)
        result: list[str] = []
        current_chunk = ""

        for part in parts:
            candidate = f"{current_chunk}{sep}{part}" if current_chunk else part
            if self._token_len(candidate) <= self.chunk_size:
                current_chunk = candidate
            else:
                if current_chunk:
                    result.append(current_chunk)
                # If this single part exceeds chunk_size, recurse with finer separators
                if self._token_len(part) > self.chunk_size:
                    result.extend(self._split_text(part, remaining_seps))
                    current_chunk = ""
                else:
                    current_chunk = part

        if current_chunk:
            result.append(current_chunk)

        return result

    def _add_overlap(self, chunks: list[str]) -> list[str]:
        if self.chunk_overlap == 0 or len(chunks) <= 1:
            return chunks

        result: list[str] = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_tokens = self._enc.encode(chunks[i - 1])
            overlap_tokens = prev_tokens[-self.chunk_overlap :]
            overlap_text = self._enc.decode(overlap_tokens)
            result.append(overlap_text + chunks[i])
        return result

    def chunk(self, text: str, metadata: dict[str, str] | None = None) -> list[Chunk]:
        raw_chunks = self._split_text(text, self.SEPARATORS)
        overlapped = self._add_overlap(raw_chunks)

        base_meta = metadata or {}
        return [
            Chunk(
                content=c.strip(),
                metadata={**base_meta, "chunk_index": i, "strategy": "recursive"},
            )
            for i, c in enumerate(overlapped)
            if c.strip()
        ]


class SemanticChunker(ChunkingStrategy):
    """Semantic chunking based on embedding similarity breakpoints.

    Groups consecutive sentences that are semantically similar,
    splitting when cosine distance exceeds a threshold.
    """

    def __init__(
        self,
        embedding_fn: callable | None = None,
        breakpoint_threshold: float = 0.3,
        max_chunk_tokens: int = settings.chunk_size,
    ) -> None:
        self._embed = embedding_fn
        self.threshold = breakpoint_threshold
        self.max_tokens = max_chunk_tokens
        self._enc = tiktoken.encoding_for_model("gpt-4o")

    @staticmethod
    def _cosine_distance(a: list[float], b: list[float]) -> float:
        import numpy as np

        a_arr, b_arr = np.array(a), np.array(b)
        cos_sim = np.dot(a_arr, b_arr) / (np.linalg.norm(a_arr) * np.linalg.norm(b_arr) + 1e-10)
        return 1.0 - float(cos_sim)

    def _split_sentences(self, text: str) -> list[str]:
        import re

        # Support both English and Japanese sentence boundaries
        sentences = re.split(r"(?<=[.!?。！？])\s+", text)
        return [s.strip() for s in sentences if s.strip()]

    def chunk(self, text: str, metadata: dict[str, str] | None = None) -> list[Chunk]:
        if self._embed is None:
            raise ValueError("embedding_fn is required for SemanticChunker")

        sentences = self._split_sentences(text)
        if not sentences:
            return []

        embeddings = [self._embed(s) for s in sentences]

        # Find breakpoints where semantic distance exceeds threshold
        groups: list[list[str]] = [[sentences[0]]]
        for i in range(1, len(sentences)):
            dist = self._cosine_distance(embeddings[i - 1], embeddings[i])
            current_group_text = " ".join(groups[-1] + [sentences[i]])
            token_count = len(self._enc.encode(current_group_text))

            if dist > self.threshold or token_count > self.max_tokens:
                groups.append([sentences[i]])
            else:
                groups[-1].append(sentences[i])

        base_meta = metadata or {}
        return [
            Chunk(
                content=" ".join(group),
                metadata={**base_meta, "chunk_index": i, "strategy": "semantic"},
            )
            for i, group in enumerate(groups)
        ]
