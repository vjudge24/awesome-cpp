"""Router agent: classifies user intent and routes to appropriate sub-agent."""

import structlog
from openai import AzureOpenAI

from src.config import settings

logger = structlog.get_logger(__name__)

ROUTER_SYSTEM_PROMPT = """You are an intent classifier. Given a user query, classify it into one of the following categories:

1. "retrieval" - The user is asking a factual question that requires looking up information from documents.
2. "calculation" - The user needs a mathematical calculation or data analysis.
3. "conversation" - The user is making casual conversation or asking about your capabilities.
4. "clarification" - The query is ambiguous and needs clarification before proceeding.

Respond with ONLY the category name, nothing else.
"""


class RouterAgent:
    """Routes queries to the appropriate processing pipeline based on intent classification."""

    VALID_INTENTS = {"retrieval", "calculation", "conversation", "clarification"}

    def __init__(self, client: AzureOpenAI | None = None) -> None:
        self._client = client or AzureOpenAI(
            api_key=settings.azure_openai.api_key,
            api_version=settings.azure_openai.api_version,
            azure_endpoint=settings.azure_openai.endpoint,
        )
        self._deployment = settings.azure_openai.chat_deployment

    def classify(self, query: str, conversation_context: str = "") -> str:
        """Classify user intent. Returns one of the valid intent categories."""
        user_content = query
        if conversation_context:
            user_content = f"Conversation so far:\n{conversation_context}\n\nLatest query: {query}"

        response = self._client.chat.completions.create(
            model=self._deployment,
            messages=[
                {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=0.0,
            max_tokens=20,
        )

        intent = (response.choices[0].message.content or "").strip().lower()

        if intent not in self.VALID_INTENTS:
            logger.warning("unknown_intent", raw_intent=intent, fallback="retrieval")
            intent = "retrieval"

        logger.info("intent_classified", query=query[:80], intent=intent)
        return intent
