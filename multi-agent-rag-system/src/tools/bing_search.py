"""Bing Search grounding tool for real-time web information.

Wraps the Bing Web Search API as a LangChain-compatible tool,
enabling agents to ground answers with up-to-date web data.

This addresses the JD requirement:
  "Bing Search grounding, etc."
"""

from __future__ import annotations

import structlog
import httpx
from langchain_core.tools import tool

from src.config import settings

logger = structlog.get_logger(__name__)

BING_SEARCH_ENDPOINT = "https://api.bing.microsoft.com/v7.0/search"


def _bing_search(
    query: str,
    api_key: str,
    count: int = 5,
    market: str = "ja-JP",
) -> list[dict[str, str]]:
    """Call the Bing Web Search API and return structured results."""
    headers = {"Ocp-Apim-Subscription-Key": api_key}
    params = {
        "q": query,
        "count": str(count),
        "mkt": market,
        "responseFilter": "Webpages",
        "textDecorations": "false",
    }

    response = httpx.get(
        BING_SEARCH_ENDPOINT,
        headers=headers,
        params=params,
        timeout=10.0,
    )
    response.raise_for_status()

    data = response.json()
    web_pages = data.get("webPages", {}).get("value", [])

    results = []
    for page in web_pages:
        results.append({
            "title": page.get("name", ""),
            "url": page.get("url", ""),
            "snippet": page.get("snippet", ""),
        })

    logger.info("bing_search_complete", query=query[:60], results=len(results))
    return results


@tool
def bing_web_search(query: str, count: int = 5) -> str:
    """Search the web using Bing for real-time information grounding.

    Use this when the user's question requires up-to-date information
    that may not be in the document knowledge base.

    Args:
        query: The search query
        count: Number of results to return (default: 5)

    Returns:
        Formatted search results with titles, URLs, and snippets
    """
    api_key = getattr(settings, "bing_search_api_key", "")
    if not api_key:
        return "Bing Search API key not configured. Cannot perform web search."

    try:
        results = _bing_search(query, api_key=api_key, count=count)
    except httpx.HTTPError as e:
        return f"Bing Search failed: {e}"

    if not results:
        return f"No web results found for: {query}"

    parts = []
    for i, r in enumerate(results, 1):
        parts.append(f"[{i}] {r['title']}\n    URL: {r['url']}\n    {r['snippet']}")

    return "\n\n".join(parts)


@tool
def bing_grounded_answer(question: str) -> str:
    """Search the web and synthesize a grounded answer from search results.

    Combines Bing Search results into context suitable for RAG augmentation.

    Args:
        question: The question to answer using web search

    Returns:
        Contextual information from web search results
    """
    api_key = getattr(settings, "bing_search_api_key", "")
    if not api_key:
        return "Bing Search API key not configured."

    try:
        results = _bing_search(question, api_key=api_key, count=3)
    except httpx.HTTPError as e:
        return f"Bing Search failed: {e}"

    if not results:
        return "No web results found."

    context_parts = []
    for r in results:
        context_parts.append(f"Source: {r['url']}\n{r['snippet']}")

    return "\n\n---\n\n".join(context_parts)
