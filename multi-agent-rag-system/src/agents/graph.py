"""LangGraph-based multi-agent orchestration.

Defines the agent state, nodes, and conditional edges for the
multi-agent RAG workflow. The graph implements:

  Router -> [Retriever -> Synthesizer] | [Conversation] | [Clarification]

Each node is an autonomous agent that modifies the shared state.
"""

from typing import Annotated, TypedDict

import structlog
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from src.agents.router import RouterAgent
from src.agents.retriever import RetrieverAgent
from src.agents.synthesizer import SynthesizerAgent
from src.rag.retriever import RetrievedDocument

logger = structlog.get_logger(__name__)


# ── State Definition ────────────────────────────────────────────────


class AgentState(TypedDict):
    """Shared state across all agents in the graph."""

    query: str
    intent: str
    messages: Annotated[list[dict[str, str]], add_messages]
    documents: list[RetrievedDocument]
    answer: str
    sources: list[str]
    quality_check: dict[str, object]
    iteration: int
    error: str | None


# ── Node Functions ──────────────────────────────────────────────────

_router = RouterAgent()
_retriever_agent = RetrieverAgent()
_synthesizer = SynthesizerAgent()


def route_node(state: AgentState) -> AgentState:
    """Classify user intent and update state."""
    query = state["query"]
    conversation = "\n".join(
        f"{m['role']}: {m['content']}" for m in state.get("messages", [])[-6:]
    )
    intent = _router.classify(query, conversation_context=conversation)
    logger.info("graph_route", intent=intent)
    return {**state, "intent": intent}


def retrieve_node(state: AgentState) -> AgentState:
    """Decompose query, retrieve relevant documents."""
    query = state["query"]
    docs = _retriever_agent.retrieve(query)
    logger.info("graph_retrieve", doc_count=len(docs))
    return {**state, "documents": docs}


def synthesize_node(state: AgentState) -> AgentState:
    """Generate answer from retrieved documents with quality check."""
    query = state["query"]
    docs = state["documents"]
    history = state.get("messages", [])

    result = _synthesizer.synthesize(query, docs, conversation_history=history)

    return {
        **state,
        "answer": result["answer"],
        "sources": result["sources"],
        "quality_check": result["quality_check"],
        "iteration": state.get("iteration", 0) + 1,
    }


def conversation_node(state: AgentState) -> AgentState:
    """Handle casual conversation without retrieval."""
    from openai import AzureOpenAI
    from src.config import settings

    client = AzureOpenAI(
        api_key=settings.azure_openai.api_key,
        api_version=settings.azure_openai.api_version,
        azure_endpoint=settings.azure_openai.endpoint,
    )

    messages = [
        {"role": "system", "content": "You are a helpful assistant. Be concise and friendly."},
        *state.get("messages", [])[-6:],
        {"role": "user", "content": state["query"]},
    ]

    response = client.chat.completions.create(
        model=settings.azure_openai.chat_deployment,
        messages=messages,
        temperature=0.7,
        max_tokens=512,
    )

    answer = response.choices[0].message.content or ""
    return {**state, "answer": answer, "sources": [], "quality_check": {}}


def clarification_node(state: AgentState) -> AgentState:
    """Ask the user for clarification."""
    return {
        **state,
        "answer": "Could you please provide more details about your question? "
        "I want to make sure I give you the most relevant information.",
        "sources": [],
        "quality_check": {},
    }


# ── Conditional Edges ───────────────────────────────────────────────


def route_by_intent(state: AgentState) -> str:
    """Route to the appropriate node based on classified intent."""
    intent = state.get("intent", "retrieval")
    if intent == "retrieval":
        return "retrieve"
    elif intent == "calculation":
        return "retrieve"  # calculations also benefit from context
    elif intent == "conversation":
        return "conversation"
    elif intent == "clarification":
        return "clarification"
    return "retrieve"


def should_retry(state: AgentState) -> str:
    """Check if synthesis quality is acceptable or needs retry."""
    qc = state.get("quality_check", {})
    iteration = state.get("iteration", 0)

    # Max 2 retries
    if iteration >= 2:
        return "end"

    if not qc.get("faithful", True) or not qc.get("relevant", True):
        logger.warning("quality_check_failed_retry", iteration=iteration, quality=qc)
        return "retrieve"

    return "end"


# ── Graph Builder ───────────────────────────────────────────────────


def build_graph() -> StateGraph:
    """Build and compile the multi-agent LangGraph workflow.

    Graph structure:
        START -> route -> [retrieve -> synthesize -> (retry?)] | conversation | clarification -> END
    """
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("route", route_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("synthesize", synthesize_node)
    graph.add_node("conversation", conversation_node)
    graph.add_node("clarification", clarification_node)

    # Set entry point
    graph.set_entry_point("route")

    # Conditional routing after intent classification
    graph.add_conditional_edges(
        "route",
        route_by_intent,
        {
            "retrieve": "retrieve",
            "conversation": "conversation",
            "clarification": "clarification",
        },
    )

    # Retrieval -> Synthesis
    graph.add_edge("retrieve", "synthesize")

    # After synthesis, check quality and potentially retry
    graph.add_conditional_edges(
        "synthesize",
        should_retry,
        {
            "retrieve": "retrieve",
            "end": END,
        },
    )

    # Terminal nodes
    graph.add_edge("conversation", END)
    graph.add_edge("clarification", END)

    return graph.compile()
