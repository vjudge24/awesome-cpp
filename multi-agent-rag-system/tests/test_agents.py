"""Tests for agent components — unit tests with mocked LLM calls."""

from unittest.mock import MagicMock, patch

import pytest

from src.agents.router import RouterAgent


class TestRouterAgent:
    """Test intent classification with mocked OpenAI client."""

    def _make_mock_response(self, content: str) -> MagicMock:
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.content = content
        return response

    def test_classify_retrieval(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._make_mock_response(
            "retrieval"
        )

        router = RouterAgent(client=mock_client)
        intent = router.classify("What is the company policy on remote work?")

        assert intent == "retrieval"
        mock_client.chat.completions.create.assert_called_once()

    def test_classify_conversation(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._make_mock_response(
            "conversation"
        )

        router = RouterAgent(client=mock_client)
        intent = router.classify("Hello, how are you?")

        assert intent == "conversation"

    def test_classify_unknown_falls_back_to_retrieval(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._make_mock_response(
            "something_invalid"
        )

        router = RouterAgent(client=mock_client)
        intent = router.classify("xyz")

        assert intent == "retrieval"

    def test_classify_with_conversation_context(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._make_mock_response(
            "clarification"
        )

        router = RouterAgent(client=mock_client)
        intent = router.classify("What about that?", conversation_context="Previous context...")

        assert intent == "clarification"
        call_args = mock_client.chat.completions.create.call_args
        user_msg = call_args.kwargs["messages"][1]["content"]
        assert "Previous context" in user_msg
