"""
The orchestrator. Every user message enters here. Flow:

  message
    │
    ▼
  is it chit-chat? ──yes──► GreetingAgent responds directly (no retrieval)
    │no
    ▼
  vector search (knowledge base)
    │
    ▼
  reflection: is evidence sufficient?
    │no                         │yes
    ▼                           ▼
  web search fallback     generate grounded answer, cite document sources
    │
    ▼
  reflection: is web evidence sufficient?
    │no                         │yes
    ▼                           ▼
  refuse ("I don't know")  generate grounded answer, cite web sources
"""

from dataclasses import dataclass, field

from google import genai

from agents.greeting import GreetingAgent
from agents.reflection import ReflectionAgent
from core.models import Answer, RetrievalResult
from retrieval.vector_search import VectorSearchTool
from retrieval.web_search import WebSearchResult, WebSearchTool

NOT_FOUND_MESSAGE = (
    "I don't have information about that in the knowledge base or the web."
)

INTENT_SYSTEM_PROMPT = """Classify the user's message as exactly one word:
- GREETING — small talk, greetings, thanks, questions about the assistant itself
- QUESTION — a real question that requires looking something up

Respond with exactly one word, nothing else."""

ANSWER_SYSTEM_PROMPT = """You are a helpful assistant. Answer the question using \
ONLY the provided evidence. Do not use outside knowledge beyond the evidence given.
Be concise and direct. Do not mention "the evidence" or "the context" explicitly \
— answer naturally, as if you know this information."""


@dataclass
class AgentAnswer:
    text: str
    answerable: bool
    doc_sources: list[RetrievalResult] = field(default_factory=list)
    web_sources: list[WebSearchResult] = field(default_factory=list)


class Supervisor:
    def __init__(
        self,
        client: genai.Client,
        vector_search: VectorSearchTool,
        web_search: WebSearchTool,
        greeting_agent: GreetingAgent,
        reflection_agent: ReflectionAgent,
        model: str = "gemini-3.1-flash-lite",
    ):
        self.client = client
        self.vector_search = vector_search
        self.web_search = web_search
        self.greeting_agent = greeting_agent
        self.reflection_agent = reflection_agent
        self.model = model

    def handle(self, message: str) -> AgentAnswer:
        if self._is_greeting(message):
            text = self.greeting_agent.respond(message)
            return AgentAnswer(text=text, answerable=True)

        # Step 1: search the knowledge base
        doc_results = self.vector_search.search(message)
        doc_evidence = self._format_doc_evidence(doc_results)
        doc_verdict = self.reflection_agent.validate(message, doc_evidence)

        if doc_verdict.sufficient:
            answer_text = self._generate(message, doc_evidence)
            return AgentAnswer(text=answer_text, answerable=True, doc_sources=doc_results)

        # Step 2: fall back to web search
        web_results = self.web_search.search(message)
        web_evidence = self._format_web_evidence(web_results)
        web_verdict = self.reflection_agent.validate(message, web_evidence)

        if web_verdict.sufficient:
            answer_text = self._generate(message, web_evidence)
            return AgentAnswer(text=answer_text, answerable=True, web_sources=web_results)

        # Step 3: nothing sufficient anywhere — refuse rather than hallucinate
        return AgentAnswer(text=NOT_FOUND_MESSAGE, answerable=False)

    def _is_greeting(self, message: str) -> bool:
        response = self.client.models.generate_content(
            model=self.model,
            contents=message,
            config={
                "system_instruction": INTENT_SYSTEM_PROMPT,
                "max_output_tokens": 20,
                "thinking_config": {"thinking_budget": 0},
            },
        )
        text = response.text or ""
        return "GREETING" in text.strip().upper()

    def _generate(self, question: str, evidence: str) -> str:
        prompt = f"""Evidence:
{evidence}

Question: {question}"""
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config={
                "system_instruction": ANSWER_SYSTEM_PROMPT,
                "max_output_tokens": 1024,
                "thinking_config": {"thinking_budget": 0},
            },
        )
        return response.text or ""

    @staticmethod
    def _format_doc_evidence(results: list[RetrievalResult]) -> str:
        blocks = []
        for i, r in enumerate(results, start=1):
            page = f", page {r.chunk.page_number}" if r.chunk.page_number is not None else ""
            blocks.append(f"[Doc {i}: {r.source_filename}{page}]\n{r.chunk.text}")
        return "\n\n".join(blocks)

    @staticmethod
    def _format_web_evidence(results: list[WebSearchResult]) -> str:
        blocks = []
        for i, r in enumerate(results, start=1):
            blocks.append(f"[Web {i}: {r.title} ({r.url})]\n{r.snippet}")
        return "\n\n".join(blocks)