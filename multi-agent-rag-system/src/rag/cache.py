"""Cache-Augmented Generation (CAG) — semantic and Redis-backed caching."""

from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np


@dataclass
class CacheEntry:
    """Single cached response."""

    query: str
    query_embedding: list[float]
    answer: str
    sources: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    hit_count: int = 0


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    va = np.asarray(a, dtype=np.float64)
    vb = np.asarray(b, dtype=np.float64)
    dot = float(np.dot(va, vb))
    norm = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if norm == 0:
        return 0.0
    return dot / norm


class SemanticCache:
    """In-memory semantic cache with cosine-similarity lookup and LRU eviction."""

    def __init__(
        self,
        similarity_threshold: float = 0.92,
        max_entries: int = 1024,
        ttl_seconds: float = 3600,
    ) -> None:
        self.similarity_threshold = similarity_threshold
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self._store: OrderedDict[str, CacheEntry] = OrderedDict()

    @staticmethod
    def _key(query: str) -> str:
        return hashlib.sha256(query.strip().lower().encode()).hexdigest()

    def lookup(
        self, query: str, query_embedding: list[float]
    ) -> CacheEntry | None:
        now = time.time()
        best_entry: CacheEntry | None = None
        best_sim = 0.0

        expired_keys: list[str] = []
        for key, entry in self._store.items():
            if now - entry.created_at > self.ttl_seconds:
                expired_keys.append(key)
                continue
            sim = _cosine_similarity(query_embedding, entry.query_embedding)
            if sim > best_sim:
                best_sim = sim
                best_entry = entry

        for k in expired_keys:
            self._store.pop(k, None)

        if best_entry is not None and best_sim >= self.similarity_threshold:
            best_entry.hit_count += 1
            # Move to end (most-recently used)
            key = self._key(best_entry.query)
            self._store.move_to_end(key)
            return best_entry
        return None

    def store(
        self,
        query: str,
        query_embedding: list[float],
        answer: str,
        sources: list[str] | None = None,
    ) -> None:
        key = self._key(query)
        self._store[key] = CacheEntry(
            query=query,
            query_embedding=query_embedding,
            answer=answer,
            sources=sources or [],
        )
        self._store.move_to_end(key)
        while len(self._store) > self.max_entries:
            self._store.popitem(last=False)

    def invalidate(self, query: str) -> bool:
        key = self._key(query)
        if key in self._store:
            del self._store[key]
            return True
        return False

    def clear(self) -> None:
        self._store.clear()

    @property
    def size(self) -> int:
        return len(self._store)
