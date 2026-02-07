"""Tests for conversation memory (in-memory mode)."""

import pytest

from src.memory.conversation import ConversationMemory


class TestConversationMemory:
    def test_add_and_retrieve(self):
        mem = ConversationMemory(session_id="test-1", use_redis=False)
        mem.add_message("user", "Hello")
        mem.add_message("assistant", "Hi there!")

        history = mem.get_history()
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"

    def test_max_turns_trimming(self):
        mem = ConversationMemory(session_id="test-2", max_turns=2, use_redis=False)
        for i in range(10):
            mem.add_message("user", f"Message {i}")
            mem.add_message("assistant", f"Response {i}")

        # max_turns=2 means max 4 messages
        history = mem.get_history()
        assert len(history) <= 4

    def test_clear(self):
        mem = ConversationMemory(session_id="test-3", use_redis=False)
        mem.add_message("user", "Hello")
        mem.clear()

        history = mem.get_history()
        assert len(history) == 0

    def test_get_summary(self):
        mem = ConversationMemory(session_id="test-4", use_redis=False)
        mem.add_message("user", "Hello")

        summary = mem.get_summary()
        assert summary["message_count"] == 1
        assert summary["total_tokens"] > 0

    def test_token_budget_limit(self):
        mem = ConversationMemory(
            session_id="test-5", max_tokens=10, use_redis=False
        )
        mem.add_message("user", "A " * 100)
        mem.add_message("assistant", "B " * 100)

        history = mem.get_history()
        # Should be limited by token budget
        total_content = " ".join(m["content"] for m in history)
        assert len(total_content) < 200 * 2  # Much less than full content

    def test_separate_sessions(self):
        mem1 = ConversationMemory(session_id="session-a", use_redis=False)
        mem2 = ConversationMemory(session_id="session-b", use_redis=False)

        mem1.add_message("user", "For session A")
        mem2.add_message("user", "For session B")

        assert len(mem1.get_history()) == 1
        assert mem1.get_history()[0]["content"] == "For session A"
        assert mem2.get_history()[0]["content"] == "For session B"
