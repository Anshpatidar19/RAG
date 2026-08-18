"""
Web Research Agent.

Flow:

    User Question
         ↓
    WebSearchTool
         ↓
    Search Results
         ↓
    fetch_page()
         ↓
    requests OR Playwright
         ↓
    Actual webpage content
         ↓
    Gemini
         ↓
    Research Answer

Web research is temporary context.

It is NOT automatically stored in Pinecone.
"""

from dataclasses import dataclass, field

from google import genai

from agents.retry_utils import (
    generate_stream_with_retry,
    generate_with_retry,
)

from ingestion.web_parser import (
    WebFetchError,
    fetch_page,
)

from retrieval.web_search import (
    WebSearchResult,
    WebSearchTool,
)


WEB_RESEARCH_SYSTEM_PROMPT = """
You are a web research assistant.

Answer the user's question using ONLY the supplied web research.

Rules:

1. Do not invent facts that are not supported by the research.
2. Prefer actual webpage content over search snippets.
3. If sources disagree, clearly mention the disagreement.
4. For current information, prefer newer sources when dates are available.
5. Give a concise but useful answer.
6. Do not mention internal implementation details.
7. Do not claim that you visited a page unless its content is included
   in the supplied research.
8. Do not use your own outside knowledge to fill missing facts.
"""


@dataclass
class ResearchContext:
    evidence: str
    sources: list[WebSearchResult] = field(
        default_factory=list
    )
    answerable: bool = False
    answer: str = ""


class WebResearchAgent:

    def __init__(
        self,
        client: genai.Client,
        web_search: WebSearchTool,
        model: str = "gemini-3.1-flash-lite",
        max_pages: int = 3,
        max_chars_per_page: int = 12000,
    ):
        self.client = client
        self.web_search = web_search
        self.model = model
        self.max_pages = max_pages
        self.max_chars_per_page = max_chars_per_page

    # ------------------------------------------------------------------
    # Prepare research
    # ------------------------------------------------------------------

    def prepare_research(
        self,
        question: str,
    ) -> ResearchContext:

        # --------------------------------------------------------------
        # 1. Search web
        # --------------------------------------------------------------

        search_results = self.web_search.search(
            question
        )

        if not search_results:

            return ResearchContext(
                evidence="",
                sources=[],
                answerable=False,
            )

        evidence_blocks = []
        successful_sources = []

        # --------------------------------------------------------------
        # 2. Visit top search results
        # --------------------------------------------------------------

        for result in search_results[: self.max_pages]:

            if not result.url:
                continue

            try:

                title, text = fetch_page(
                    result.url
                )

            except WebFetchError:
                # One website failing should not kill the entire
                # research process.
                continue

            if not text.strip():
                continue

            source_number = (
                len(successful_sources) + 1
            )

            evidence_blocks.append(
                f"""
[Source {source_number}]
Title: {title or result.title}
URL: {result.url}

Content:
{text[:self.max_chars_per_page]}
""".strip()
            )

            successful_sources.append(
                WebSearchResult(
                    title=title or result.title,
                    url=result.url,
                    snippet=result.snippet,
                    score=result.score,
                )
            )

        # --------------------------------------------------------------
        # 3. If webpages couldn't be fetched, use search snippets
        # as a last-resort fallback.
        # --------------------------------------------------------------

        if not evidence_blocks:

            for result in search_results:

                evidence_blocks.append(
                    f"""
[Search Result]
Title: {result.title}
URL: {result.url}

Snippet:
{result.snippet}
""".strip()
                )

            successful_sources = search_results

        evidence = "\n\n".join(
            evidence_blocks
        )

        return ResearchContext(
            evidence=evidence,
            sources=successful_sources,
            answerable=bool(evidence.strip()),
        )

    # ------------------------------------------------------------------
    # Non-streaming research
    # ------------------------------------------------------------------

    def research(
        self,
        question: str,
    ) -> ResearchContext:

        context = self.prepare_research(
            question
        )

        if not context.answerable:
            return context

        answer = self.generate(
            question,
            context.evidence,
        )

        # Store the generated answer in the same object dynamically.
        # This keeps compatibility with Supervisor's existing
        # AgentAnswer handling.
        context.answer = answer

        return context

    # ------------------------------------------------------------------
    # Generate final answer
    # ------------------------------------------------------------------

    def generate(
        self,
        question: str,
        evidence: str,
    ) -> str:

        prompt = f"""
User question:

{question}

Web research:

{evidence}
"""

        response = generate_with_retry(
            self.client,
            model=self.model,
            contents=prompt,
            config={
                "system_instruction": WEB_RESEARCH_SYSTEM_PROMPT,
                "max_output_tokens": 1500,
                "thinking_config": {
                    "thinking_budget": 0
                },
            },
        )

        return (
            response.text or ""
        ).strip()

    # ------------------------------------------------------------------
    # Streaming final answer
    # ------------------------------------------------------------------

    def generate_stream(
        self,
        question: str,
        evidence: str,
    ):

        prompt = f"""
User question:

{question}

Web research:

{evidence}
"""

        stream = generate_stream_with_retry(
            self.client,
            model=self.model,
            contents=prompt,
            config={
                "system_instruction": WEB_RESEARCH_SYSTEM_PROMPT,
                "max_output_tokens": 1500,
                "thinking_config": {
                    "thinking_budget": 0
                },
            },
        )

        for chunk in stream:

            if chunk.text:
                yield chunk.text

    # ------------------------------------------------------------------
    # Compatibility streaming method
    # ------------------------------------------------------------------

    def research_stream(
        self,
        question: str,
    ):

        context = self.prepare_research(
            question
        )

        if not context.answerable:

            yield "I could not find useful information on the web."
            return

        for token in self.generate_stream(
            question,
            context.evidence,
        ):
            yield token