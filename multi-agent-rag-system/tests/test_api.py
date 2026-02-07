"""Integration tests for the FastAPI application.

Tests run against the real FastAPI app using TestClient,
with LangGraph/Azure services mocked at the boundary.
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.main import app


@pytest.fixture
def client():
    """Create a test client with mocked agent graph."""
    mock_graph = MagicMock()
    mock_graph.invoke.return_value = {
        "answer": "This is a test answer based on the context.",
        "sources": ["test_doc.pdf"],
        "quality_check": {"faithful": True, "relevant": True, "complete": True},
        "iteration": 1,
    }

    with patch("src.main.agent_graph", mock_graph):
        yield TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_ok(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "environment" in data


class TestQueryEndpoint:
    def test_query_returns_answer(self, client):
        response = client.post(
            "/query",
            json={"question": "What is RAG?"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data
        assert "quality_check" in data
        assert "metrics" in data

    def test_query_with_session_id(self, client):
        response = client.post(
            "/query",
            json={"question": "Tell me more", "session_id": "test-session-42"},
        )
        assert response.status_code == 200
        assert response.json()["session_id"] == "test-session-42"

    def test_query_empty_question(self, client):
        response = client.post(
            "/query",
            json={"question": ""},
        )
        assert response.status_code == 200


class TestMetricsEndpoint:
    def test_metrics_returns_summary(self, client):
        response = client.get("/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "total_requests" in data


class TestSessionEndpoint:
    def test_clear_session(self, client):
        response = client.delete("/session/test-session-123")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "cleared"
        assert data["session_id"] == "test-session-123"


class TestPromptsEndpoint:
    def test_list_prompts(self, client):
        response = client.get("/prompts")
        assert response.status_code == 200
        data = response.json()
        assert "router_system" in data
        assert "synthesizer_system" in data
        assert "quality_check" in data
        assert "query_decomposition" in data

    def test_get_prompt_by_name(self, client):
        response = client.get("/prompts/router_system")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "router_system"
        assert data["version"] == "1.0.0"
        assert "template" in data
        assert "content_hash" in data

    def test_get_prompt_not_found(self, client):
        response = client.get("/prompts/nonexistent_prompt")
        assert response.status_code == 404
