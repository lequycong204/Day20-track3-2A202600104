"""Search client abstraction for ResearcherAgent.

Production note: agents should depend on this interface instead of importing
a search SDK directly. Use MockSearchClient for local dev/tests.
"""

from __future__ import annotations

import logging

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import SourceDocument

logger = logging.getLogger(__name__)


class SearchClient:
    """Provider-agnostic search client backed by Tavily.

    Initialises the Tavily connection once in __init__ so that search()
    stays focused on a single responsibility.
    """

    def __init__(self) -> None:
        settings = get_settings()
        if not settings.tavily_api_key:
            raise ValueError(
                "TAVILY_API_KEY is not set. "
                "Add it to your .env file or set the environment variable. "
                "If you want a zero-dependency fallback, use MockSearchClient instead."
            )
        try:
            from tavily import TavilyClient  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "tavily-python package is not installed. "
                "Run: pip install tavily-python"
            ) from exc

        self._client = TavilyClient(api_key=settings.tavily_api_key)
        logger.debug("SearchClient initialised with Tavily")

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        """Search the web for documents relevant to a query.

        Args:
            query: The search query string.
            max_results: Maximum number of results to return (1-20).

        Returns:
            A list of SourceDocument objects with title, url, and snippet.
        """
        logger.debug("SearchClient.search | query=%r max_results=%d", query, max_results)

        response = self._client.search(
            query=query,
            max_results=max_results,
            search_depth="advanced",
            include_answer=False,
        )

        results: list[SourceDocument] = []
        for item in response.get("results", []):
            results.append(
                SourceDocument(
                    title=item.get("title", "Untitled"),
                    url=item.get("url"),
                    snippet=item.get("content", ""),
                    metadata={
                        "score": item.get("score"),
                        "published_date": item.get("published_date"),
                    },
                )
            )

        logger.info(
            "SearchClient.search | query=%r returned %d results",
            query,
            len(results),
        )
        return results