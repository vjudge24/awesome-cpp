"""Conversation memory management with Redis backend.

Provides sliding-window and token-based memory management for
multi-turn conversations. Supports both in-memory and Redis backends.
"""

import json
import time

import structlog
import tiktoken

from src.config import settings

logger = structlog.get_logger(__name__)


class ConversationMemory:
    """Manages conversation history per session with configurable backends.

    Supports:
    - Sliding window: keep last N message pairs
    - Token budget: keep messages within a token limit
    - Redis persistence: survive server restarts
    - In-memory fallback: for development/testing
    """

    def __init__(
        self,
        session_id: str,
        max_turns: int = 20,
        max_tokens: int = 4096,
        use_redis: bool = True,
        ttl_seconds: int = 3600,
    ) -> None:
        self._session_id = session_id
        self._max_turns = max_turns
        self._max_tokens = max_tokens
        self._ttl = ttl_seconds
        self._enc = tiktoken.encoding_for_model("gpt-4o")
        self._redis = None
        self._local_store: list[dict[str, str]] = []

        if use_redis:
            try:
                import redis

                self._redis = redis.from_url(settings.redis_url, decode_responses=True)
                self._redis.ping()
                logger.info("memory_redis_connected", session_id=session_id)
            except Exception as e:
                logger.warning("memory_redis_fallback", error=str(e))
                self._redis = None

    @property
    def _key(self) -> str:
        return f"conversation:{self._session_id}"

    def add_message(self, role: str, content: str) -> None:
        """Add a message to the conversation history."""
        message = {"role": role, "content": content, "timestamp": str(time.time())}

        if self._redis:
            self._redis.rpush(self._key, json.dumps(message))
            self._redis.expire(self._key, self._ttl)
        else:
            self._local_store.append(message)

        self._trim()

    def get_history(self) -> list[dict[str, str]]:
        """Retrieve conversation history within token budget."""
        messages = self._load_messages()

        # Apply token budget from most recent
        result: list[dict[str, str]] = []
        token_count = 0

        for msg in reversed(messages):
            msg_tokens = len(self._enc.encode(msg["content"]))
            if token_count + msg_tokens > self._max_tokens:
                break
            result.insert(0, {"role": msg["role"], "content": msg["content"]})
            token_count += msg_tokens

        return result

    def clear(self) -> None:
        """Clear all messages for this session."""
        if self._redis:
            self._redis.delete(self._key)
        else:
            self._local_store.clear()
        logger.info("memory_cleared", session_id=self._session_id)

    def get_summary(self) -> dict[str, int]:
        """Get memory usage statistics."""
        messages = self._load_messages()
        total_tokens = sum(len(self._enc.encode(m["content"])) for m in messages)
        return {
            "session_id_hash": hash(self._session_id) % 10000,
            "message_count": len(messages),
            "total_tokens": total_tokens,
            "max_turns": self._max_turns,
            "max_tokens": self._max_tokens,
        }

    def _load_messages(self) -> list[dict[str, str]]:
        if self._redis:
            raw = self._redis.lrange(self._key, 0, -1)
            return [json.loads(r) for r in raw]
        return list(self._local_store)

    def _trim(self) -> None:
        """Trim to max_turns (pairs = max_turns * 2 messages)."""
        max_messages = self._max_turns * 2

        if self._redis:
            length = self._redis.llen(self._key)
            if length > max_messages:
                self._redis.ltrim(self._key, length - max_messages, -1)
        else:
            if len(self._local_store) > max_messages:
                self._local_store = self._local_store[-max_messages:]
