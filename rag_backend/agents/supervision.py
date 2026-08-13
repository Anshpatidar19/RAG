"""
The Supervisor Agent.

Flow:

message
    │
    ▼
is it chit-chat?
    │ yes
    ▼
GreetingAgent responds directly
    │
    │ no
    ▼
vector search
    │
    ▼
retrieval score threshold
    │
    ├── insufficient ──► web search fallback
    │
    └── sufficient
            │
            ▼
      Reflection Agent
            │
       ┌────┴────┐
       │         │
      yes        no
       │         │
       ▼         ▼
   generate    web search
    answer         │
                   ▼
             Reflection Agent
                   │
              ┌────┴────┐
              │         │
             yes        no
              │         │
              ▼         ▼
          generate    refuse
           answer
"""


from dataclasses import dataclass, field

from google import genai

from agents.greeting import GreetingAgent
from agents.reflection import ReflectionAgent
from core.models import RetrievalResult
from retrieval.retriever import Retriever
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
        retriever: Retriever,
        model: str = "gemini-3.1-flash-lite",
    ):
        self.client = client
        self.vector_search = vector_search
        self.web_search = web_search
        self.greeting_agent = greeting_agent
        self.reflection_agent = reflection_agent
        self.retriever = retriever
        self.model = model

    def handle(self, message: str) -> AgentAnswer:

        # ---------------------------------------
        # 1. Greeting / casual query
        # ---------------------------------------

        if self._is_greeting(message):

            text = self.greeting_agent.respond(message)

            return AgentAnswer(
                text=text,
                answerable=True,
            )

        # ---------------------------------------
        # 2. Search knowledge base
        # ---------------------------------------

        doc_results = self.vector_search.search(message)

        # ---------------------------------------
        # 3. Check retrieval score threshold
        # ---------------------------------------

        if not self.retriever.has_sufficient_context(
            doc_results
        ):

            # Retrieval result is too weak.
            # Do NOT send weak evidence to Reflection Agent.
            doc_results = []

        else:

            # ---------------------------------------
            # 4. Reflection Agent
            # ---------------------------------------

            doc_evidence = self._format_doc_evidence(
                doc_results
            )

            doc_verdict = self.reflection_agent.validate(
                message,
                doc_evidence,
            )

            # ---------------------------------------
            # 5. Document answer
            # ---------------------------------------

            if doc_verdict.sufficient:

                answer_text = self._generate(
                    message,
                    doc_evidence,
                )

                return AgentAnswer(
                    text=answer_text,
                    answerable=True,
                    doc_sources=doc_results,
                )

            # Reflection says retrieved content
            # is not sufficient.
            doc_results = []

        # ---------------------------------------
        # 6. Web search fallback
        # ---------------------------------------

        web_results = self.web_search.search(
            message
        )

        web_evidence = self._format_web_evidence(
            web_results
        )

        # ---------------------------------------
        # 7. Reflection Agent for web evidence
        # ---------------------------------------

        web_verdict = self.reflection_agent.validate(
            message,
            web_evidence,
        )

        # ---------------------------------------
        # 8. Web-grounded answer
        # ---------------------------------------

        if web_verdict.sufficient:

            answer_text = self._generate(
                message,
                web_evidence,
            )

            return AgentAnswer(
                text=answer_text,
                answerable=True,
                web_sources=web_results,
            )

        # ---------------------------------------
        # 9. Nothing sufficient
        # ---------------------------------------

        return AgentAnswer(
            text=NOT_FOUND_MESSAGE,
            answerable=False,
        )

    def _is_greeting(
        self,
        message: str,
    ) -> bool:

        response = self.client.models.generate_content(
            model=self.model,
            contents=message,
            config={
                "system_instruction": INTENT_SYSTEM_PROMPT,
                "max_output_tokens": 20,
                "thinking_config": {
                    "thinking_budget": 0
                },
            },
        )

        text = response.text or ""

        return "GREETING" in text.strip().upper()

    def _generate(
        self,
        question: str,
        evidence: str,
    ) -> str:

        prompt = f"""Evidence:
{evidence}

Question: {question}"""

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config={
                "system_instruction": ANSWER_SYSTEM_PROMPT,
                "max_output_tokens": 1024,
                "thinking_config": {
                    "thinking_budget": 0
                },
            },
        )

        return response.text or ""

    @staticmethod
    def _format_doc_evidence(
        results: list[RetrievalResult],
    ) -> str:

        blocks = []

        for i, r in enumerate(
            results,
            start=1,
        ):

            page = (
                f", page {r.chunk.page_number}"
                if r.chunk.page_number is not None
                else ""
            )

            blocks.append(
                f"[Doc {i}: "
                f"{r.source_filename}"
                f"{page}]\n"
                f"{r.chunk.text}"
            )

        return "\n\n".join(blocks)

    @staticmethod
    def _format_web_evidence(
        results: list[WebSearchResult],
    ) -> str:

        blocks = []

        for i, r in enumerate(
            results,
            start=1,
        ):

            blocks.append(
                f"[Web {i}: "
                f"{r.title} "
                f"({r.url})]\n"
                f"{r.snippet}"
            )

        return "\n\n".join(blocks)