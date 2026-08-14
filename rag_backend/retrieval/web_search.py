"""
Web search fallback tool, used by the supervisor when knowledge-base
retrieval doesn't produce sufficient evidence to answer a question.
Uses ddgs (DuckDuckGo search) — free, no API key required.
"""

from dataclasses import dataclass

from ddgs import DDGS


@dataclass
class WebSearchResult:
    title: str
    url: str
    snippet: str
    score: float = 0.0  # DuckDuckGo doesn't return a relevance score; kept for interface parity


class WebSearchTool:
    name = "search_web"
    description = (
        "Searches the public web for information not found in the user's uploaded "
        "documents. Use only when knowledge-base search does not return sufficient "
        "evidence to answer the question."
    )

    def __init__(self, max_results: int = 3):
        self.max_results = max_results

    def search(self, query: str) -> list[WebSearchResult]:
        with DDGS() as ddgs:
            raw_results = ddgs.text(query, max_results=self.max_results)

        results = []
        for item in raw_results:
            results.append(
                WebSearchResult(
                    title=item.get("title", ""),
                    url=item.get("href", ""),
                    snippet=item.get("body", ""),
                )
            )
        return results