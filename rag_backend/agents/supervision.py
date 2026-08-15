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

handle_stream() follows the exact same routing as handle() above, but
yields progress/token events instead of returning one final object —
used by the /ask/stream endpoint for live status updates + token
streaming in the UI.
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


STANDALONE_QUERY_PROMPT = """You rewrite a follow-up question into a standalone \
question using the conversation history for context, so it can be searched on its \
own without needing the prior turns.

Rules:
- If the follow-up is already standalone (doesn't depend on prior turns), return it unchanged.
- Resolve pronouns and vague references ("it", "that", "the second one") using the history.
- Respond with ONLY the rewritten question, nothing else — no preamble, no quotes."""


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

    # ------------------------------------------------------------
    # Non-streaming path (used by /ask)
    # ------------------------------------------------------------

    def handle(self, message: str, history: list[dict] | None = None) -> AgentAnswer:

        # Check greeting on the RAW message first — rewriting into a
        # standalone query using history can turn "hi"/"thanks" into
        # something that no longer looks like a greeting, which was
        # causing greetings to be misrouted into document search once
        # a conversation had any history.
        if self._is_greeting(message):
            text = self.greeting_agent.respond(message)
            return AgentAnswer(text=text, answerable=True)

        search_query = self._standalone_query(message, history)

        # 2. Search knowledge base
        doc_results = self.vector_search.search(search_query)

        # 3. Check retrieval score threshold (cheap gate before spending
        #    an LLM call on reflection)
        if not self.retriever.has_sufficient_context(doc_results):
            doc_results = []
        else:
            # 4. Reflection Agent judges relevance, not just similarity
            doc_evidence = self._format_doc_evidence(doc_results)
            doc_verdict = self.reflection_agent.validate(search_query, doc_evidence)

            if doc_verdict.sufficient:
                answer_text = self._generate(search_query, doc_evidence)
                return AgentAnswer(
                    text=answer_text,
                    answerable=True,
                    doc_sources=doc_results,
                )

            doc_results = []

        # 6. Web search fallback
        web_results = self.web_search.search(search_query)
        web_evidence = self._format_web_evidence(web_results)

        # 7. Reflection Agent for web evidence
        web_verdict = self.reflection_agent.validate(search_query, web_evidence)

        # 8. Web-grounded answer
        if web_verdict.sufficient:
            answer_text = self._generate(search_query, web_evidence)
            return AgentAnswer(
                text=answer_text,
                answerable=True,
                web_sources=web_results,
            )

        # 9. Nothing sufficient
        return AgentAnswer(text=NOT_FOUND_MESSAGE, answerable=False)

    # ------------------------------------------------------------
    # Streaming path (used by /ask/stream)
    # ------------------------------------------------------------

    def handle_stream(self, message: str, history: list[dict] | None = None):
        """
        Same routing logic as handle(), but yields progress events and
        streamed answer tokens instead of returning one final AgentAnswer.
        Event shapes:
          {"type": "status", "message": str}
          {"type": "token", "text": str}
          {"type": "sources", "doc_sources": [...], "web_sources": [...], "answerable": bool}
          {"type": "done"}
        """

        yield {"type": "status", "message": "Reading your message..."}

        # Check greeting on the RAW message first, before any history-based
        # rewrite — same reasoning as handle() above.
        if self._is_greeting(message):
            yield {"type": "status", "message": "Responding..."}
            for token in self.greeting_agent.respond_stream(message):
                yield {"type": "token", "text": token}
            yield {"type": "sources", "doc_sources": [], "web_sources": [], "answerable": True}
            yield {"type": "done"}
            return

        if history:
            yield {"type": "status", "message": "Understanding your question..."}
        search_query = self._standalone_query(message, history)

        yield {"type": "status", "message": "Searching your documents..."}
        doc_results = self.vector_search.search(search_query)

        if self.retriever.has_sufficient_context(doc_results):
            yield {"type": "status", "message": "Checking if the results answer your question..."}
            doc_evidence = self._format_doc_evidence(doc_results)
            doc_verdict = self.reflection_agent.validate(search_query, doc_evidence)

            if doc_verdict.sufficient:
                yield {"type": "status", "message": "Writing answer from your documents..."}
                for token in self._generate_stream(search_query, doc_evidence):
                    yield {"type": "token", "text": token}
                yield {
                    "type": "sources",
                    "doc_sources": doc_results,
                    "web_sources": [],
                    "answerable": True,
                }
                yield {"type": "done"}
                return

        yield {"type": "status", "message": "Searching the web..."}
        web_results = self.web_search.search(search_query)
        web_evidence = self._format_web_evidence(web_results)

        yield {"type": "status", "message": "Checking if the web results answer your question..."}
        web_verdict = self.reflection_agent.validate(search_query, web_evidence)

        if web_verdict.sufficient:
            yield {"type": "status", "message": "Writing answer from web results..."}
            for token in self._generate_stream(search_query, web_evidence):
                yield {"type": "token", "text": token}
            yield {
                "type": "sources",
                "doc_sources": [],
                "web_sources": web_results,
                "answerable": True,
            }
            yield {"type": "done"}
            return

        yield {"type": "token", "text": NOT_FOUND_MESSAGE}
        yield {"type": "sources", "doc_sources": [], "web_sources": [], "answerable": False}
        yield {"type": "done"}

    # ------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------

    def _standalone_query(self, message: str, history: list[dict] | None) -> str:
        """
        Rewrites a follow-up question into a standalone question using
        recent conversation history, so retrieval can search on it
        directly. Skipped entirely (no LLM call) when there's no history,
        since a first question is always already standalone.
        """
        if not history:
            return message

        history_text = "\n".join(
            f"User: {turn['question']}\nAssistant: {turn['answer']}"
            for turn in history[-6:]
        )
        prompt = f"""Conversation history:
{history_text}

Follow-up question: {message}"""

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config={
                "system_instruction": STANDALONE_QUERY_PROMPT,
                "max_output_tokens": 150,
                "thinking_config": {"thinking_budget": 0},
            },
        )
        rewritten = (response.text or "").strip()
        return rewritten or message

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

    def _generate_stream(self, question: str, evidence: str):
        prompt = f"""Evidence:
{evidence}

Question: {question}"""

        stream = self.client.models.generate_content_stream(
            model=self.model,
            contents=prompt,
            config={
                "system_instruction": ANSWER_SYSTEM_PROMPT,
                "max_output_tokens": 1024,
                "thinking_config": {"thinking_budget": 0},
            },
        )
        for chunk in stream:
            if chunk.text:
                yield chunk.text

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