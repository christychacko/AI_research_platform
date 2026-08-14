"""
tools/search_tool.py

Free web search via DuckDuckGo (no API key, no signup, no cost).
Exposed as a plain LangChain @tool so it can be handed to LangGraph nodes
directly, and also wrapped for CrewAI / AutoGen further down the stack.
"""

from langchain_core.tools import tool
from ddgs import DDGS


@tool
def web_search(query: str, max_results: int = 5) -> str:
    """
    Search the web for current information on a topic.
    Use this when you need recent facts, news, or general research material
    that isn't already in the local knowledge base.

    Args:
        query: the search query string
        max_results: how many results to return (default 5)
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
    except Exception as e:
        return f"Search failed: {e}"

    if not results:
        return "No results found."

    formatted = []
    for i, r in enumerate(results, 1):
        formatted.append(
            f"{i}. {r.get('title', 'No title')}\n"
            f"   URL: {r.get('href', 'N/A')}\n"
            f"   Snippet: {r.get('body', 'N/A')}"
        )
    return "\n\n".join(formatted)


if __name__ == "__main__":
    # quick manual test: python tools/search_tool.py
    print(web_search.invoke({"query": "latest trends in AI agents 2026", "max_results": 3}))
