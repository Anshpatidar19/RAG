"""
The Supervisor Agent.

Routing flow:

                          USER MESSAGE
                              │
                              ▼
                        Intent Classification
                              │
               ┌───────────────┼────────────────┐
               │               │                │
               ▼               ▼                ▼
           GREETING          EMAIL           QUESTION
               │               │                │
               ▼               ▼                ▼
        GreetingAgent    EmailWriterAgent   Document Search
                                               │
                                               ▼
                                       Retrieval Threshold
                                               │
                                   ┌────────────┴────────────┐
                                   │                         │
                               Sufficient               Insufficient
                                   │                         │
                                   ▼                         │
                            Reflection Agent                │
                                   │                         │
                          ┌────────┴────────┐                │
                          │                 │                │
                      Sufficient        Insufficient         │
                          │                 │                │
                          ▼                 └────────┬───────┘
                     Document RAG                    │
                                                    ▼
                                           WebResearchAgent
                                                    │
                                       ┌─────────────┴─────────────┐
                                       │                           │
                                 Web Search                  Playwright
                                       │                           │
                                       └─────────────┬─────────────┘
                                                     ▼
                                                Web Evidence
                                                     │
                                                     ▼
                                              Reflection Agent
                                                     │
                                           ┌────────┴────────┐
                                           │                 │
                                       Sufficient       Insufficient
                                           │                 │
                                           ▼                 ▼
                                      Web Answer         Refuse

The document-first behavior is intentional:
if a sufficiently relevant uploaded-document chunk can answer the
question, the web is NOT consulted.
"""

from dataclasses import dataclass, field

from google import genai

from agents.email_writer import EmailWriterAgent
from agents.greeting import GreetingAgent
from agents.reflection import ReflectionAgent
from agents.retry_utils import (
    generate_stream_with_retry,
    generate_with_retry,
)
from agents.web_research import WebResearchAgent

from core.models import RetrievalResult

from retrieval.retriever import Retriever
from retrieval.vector_search import VectorSearchTool
from retrieval.web_search import WebSearchResult, WebSearchTool


NOT_FOUND_MESSAGE = (
    "I don't have enough information about that in the knowledge base "
    "or the web."
)


# ---------------------------------------------------------------------------
# Intent classification
# ---------------------------------------------------------------------------

INTENT_SYSTEM_PROMPT = """
Classify the user's message as exactly one word.

Possible intents:

- GREETING — greetings, small talk, thanks, casual conversation,
  or questions about the assistant itself.

- EMAIL — the user wants you to write, draft, rewrite, improve,
  format, or compose an email or professional message.

- QUESTION — a real question that requires information retrieval,
  document search, web research, analysis, or another
  knowledge-based task.

Respond with exactly one word:

GREETING
EMAIL
QUESTION

Do not return anything else.
"""


# ---------------------------------------------------------------------------
# Answer generation
# ---------------------------------------------------------------------------

ANSWER_SYSTEM_PROMPT = """
You are a helpful assistant.

Answer the question using ONLY the facts present in the provided evidence.

Do not bring in outside knowledge that the evidence does not support.

You MAY perform straightforward computation or synthesis over the evidence
itself — for example:
- reading a percentage from a table
- summing figures
- comparing values
- combining two facts stated in the evidence

That counts as answering FROM the evidence, not as outside knowledge.

Be concise and direct.

Do not mention "the evidence" or "the context" explicitly.
Answer naturally, as if you know this information.
"""


# ---------------------------------------------------------------------------
# Standalone query rewriting
# ---------------------------------------------------------------------------

STANDALONE_QUERY_PROMPT = """
You rewrite a follow-up question into a standalone question using the
conversation history for context, so it can be searched on its own.

Rules:

- If the follow-up is already standalone and does not depend on prior
  turns, return it unchanged.
- Resolve pronouns and vague references such as:
  "it", "that", "the second one", "this", etc.
  using the conversation history.
- Preserve the user's actual intent.
- Do not add information that is not present in the conversation.
- Respond with ONLY the rewritten question.
- No preamble.
- No quotes.
"""


# ---------------------------------------------------------------------------
# Response model
# ---------------------------------------------------------------------------

@dataclass
class AgentAnswer:
    text: str
    answerable: bool
    doc_sources: list[RetrievalResult] = field(default_factory=list)
    web_sources: list[WebSearchResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Supervisor
# ---------------------------------------------------------------------------

class Supervisor:

    def __init__(
        self,
        client: genai.Client,
        vector_search: VectorSearchTool,
        web_search: WebSearchTool,
        greeting_agent: GreetingAgent,
        reflection_agent: ReflectionAgent,
        retriever: Retriever,
        email_agent: EmailWriterAgent,
        web_research_agent: WebResearchAgent,
        model: str = "gemini-3.1-flash-lite",
    ):
        self.client = client
        self.vector_search = vector_search

        # Kept for compatibility with the existing architecture.
        # Actual web research is now handled by WebResearchAgent.
        self.web_search = web_search

        self.greeting_agent = greeting_agent
        self.reflection_agent = reflection_agent
        self.retriever = retriever

        # New specialized agents.
        self.email_agent = email_agent
        self.web_research_agent = web_research_agent

        self.model = model

    # -----------------------------------------------------------------------
    # Non-streaming path
    # Used by /ask
    # -----------------------------------------------------------------------

    def handle(
        self,
        message: str,
        history: list[dict] | None = None,
    ) -> AgentAnswer:

        # ---------------------------------------------------------------
        # 1. Classify the user's intent.
        #
        # This happens BEFORE standalone query rewriting because an email
        # request should not accidentally be converted into a document
        # retrieval query.
        # ---------------------------------------------------------------

        intent = self._classify_intent(message)

        # ---------------------------------------------------------------
        # 2. Greeting
        # ---------------------------------------------------------------

        if intent == "GREETING":

            text = self.greeting_agent.respond(message)

            return AgentAnswer(
                text=text,
                answerable=True,
            )

        # ---------------------------------------------------------------
        # 3. Email Writing Agent
        # ---------------------------------------------------------------

        if intent == "EMAIL":

            text = self.email_agent.write(message)

            return AgentAnswer(
                text=text,
                answerable=True,
            )

        # ---------------------------------------------------------------
        # 4. Normal question
        # ---------------------------------------------------------------

        search_query = self._standalone_query(
            message,
            history,
        )

        # ---------------------------------------------------------------
        # 5. Search user's knowledge base FIRST
        # ---------------------------------------------------------------

        doc_results = self.vector_search.search(
            search_query
        )

        # ---------------------------------------------------------------
        # 6. Retrieval score threshold
        #
        # Cheap gate before calling ReflectionAgent.
        # ---------------------------------------------------------------

        if self.retriever.has_sufficient_context(
            doc_results
        ):

            # -----------------------------------------------------------
            # 7. Reflection Agent checks whether the chunks actually
            # answer the question.
            # -----------------------------------------------------------

            doc_evidence = self._format_doc_evidence(
                doc_results
            )

            # ===========================================================
            # REFLECTION DEBUG
            # ===========================================================

            self._debug_reflection_context(
                search_query,
                doc_results,
                doc_evidence,
            )

            # ===========================================================
            # END REFLECTION DEBUG
            # ===========================================================

            doc_verdict = self.reflection_agent.validate(
                search_query,
                doc_evidence,
            )

            # -----------------------------------------------------------
            # 8. Strong document answer → STOP.
            #
            # IMPORTANT:
            # We DO NOT call the web here.
            # -----------------------------------------------------------

            if doc_verdict.sufficient:

                answer_text = self._generate(
                    search_query,
                    doc_evidence,
                )

                return AgentAnswer(
                    text=answer_text,
                    answerable=True,
                    doc_sources=doc_results,
                )

        # ---------------------------------------------------------------
        # 9. No sufficiently useful document evidence.
        #
        # Now and ONLY now we use WebResearchAgent.
        # ---------------------------------------------------------------

        research = self.web_research_agent.research(
            search_query
        )

        # ---------------------------------------------------------------
        # 10. Web research successfully answered the question.
        # ---------------------------------------------------------------

        if research.answerable:

            return AgentAnswer(
                text=research.answer,
                answerable=True,
                web_sources=research.sources,
            )

        # ---------------------------------------------------------------
        # 11. Nothing was sufficient.
        # ---------------------------------------------------------------

        return AgentAnswer(
            text=NOT_FOUND_MESSAGE,
            answerable=False,
        )

    # -----------------------------------------------------------------------
    # Streaming path
    # Used by /ask/stream
    # -----------------------------------------------------------------------

    def handle_stream(
        self,
        message: str,
        history: list[dict] | None = None,
    ):
        """
        Same routing logic as handle(), but yields progress events
        and streamed answer tokens.

        Event shapes:

            {"type": "status", "message": str}

            {"type": "token", "text": str}

            {
                "type": "sources",
                "doc_sources": [...],
                "web_sources": [...],
                "answerable": bool
            }

            {"type": "done"}
        """

        # ---------------------------------------------------------------
        # 1. Read message
        # ---------------------------------------------------------------

        yield {
            "type": "status",
            "message": "Understanding your request...",
        }

        intent = self._classify_intent(
            message
        )

        # ---------------------------------------------------------------
        # 2. Greeting
        # ---------------------------------------------------------------

        if intent == "GREETING":

            yield {
                "type": "status",
                "message": "Responding...",
            }

            for token in self.greeting_agent.respond_stream(
                message
            ):
                yield {
                    "type": "token",
                    "text": token,
                }

            yield {
                "type": "sources",
                "doc_sources": [],
                "web_sources": [],
                "answerable": True,
            }

            yield {
                "type": "done"
            }

            return

        # ---------------------------------------------------------------
        # 3. Email writing
        # ---------------------------------------------------------------

        if intent == "EMAIL":

            yield {
                "type": "status",
                "message": "Writing your email...",
            }

            for token in self.email_agent.write_stream(
                message
            ):
                yield {
                    "type": "token",
                    "text": token,
                }

            yield {
                "type": "sources",
                "doc_sources": [],
                "web_sources": [],
                "answerable": True,
            }

            yield {
                "type": "done"
            }

            return

        # ---------------------------------------------------------------
        # 4. Standalone query
        # ---------------------------------------------------------------

        if history:

            yield {
                "type": "status",
                "message": "Understanding your question...",
            }

        search_query = self._standalone_query(
            message,
            history,
        )

        # ---------------------------------------------------------------
        # 5. Search documents
        # ---------------------------------------------------------------

        yield {
            "type": "status",
            "message": "Searching your documents...",
        }

        doc_results = self.vector_search.search(
            search_query
        )

        # ---------------------------------------------------------------
        # 6. Check document relevance
        # ---------------------------------------------------------------

        if self.retriever.has_sufficient_context(
            doc_results
        ):

            yield {
                "type": "status",
                "message": (
                    "Checking whether your documents "
                    "can answer the question..."
                ),
            }

            doc_evidence = self._format_doc_evidence(
                doc_results
            )

            # ===========================================================
            # REFLECTION DEBUG
            #
            # This is the important part for your current problem.
            # It prints exactly what is sent to ReflectionAgent.
            # ===========================================================

            self._debug_reflection_context(
                search_query,
                doc_results,
                doc_evidence,
            )

            # ===========================================================
            # END REFLECTION DEBUG
            # ===========================================================

            doc_verdict = self.reflection_agent.validate(
                search_query,
                doc_evidence,
            )

            # -----------------------------------------------------------
            # 7. Document is sufficient.
            #
            # DO NOT use web.
            # -----------------------------------------------------------

            if doc_verdict.sufficient:

                yield {
                    "type": "status",
                    "message": (
                        "Writing answer from your documents..."
                    ),
                }

                for token in self._generate_stream(
                    search_query,
                    doc_evidence,
                ):
                    yield {
                        "type": "token",
                        "text": token,
                    }

                yield {
                    "type": "sources",
                    "doc_sources": doc_results,
                    "web_sources": [],
                    "answerable": True,
                }

                yield {
                    "type": "done"
                }

                return

        # ---------------------------------------------------------------
        # 8. Document insufficient → Web Research Agent
        # ---------------------------------------------------------------

        yield {
            "type": "status",
            "message": "Researching the web...",
        }

        # Search and webpage extraction happen before streaming
        # the final answer.
        research_context = self.web_research_agent.prepare_research(
            search_query
        )

        if not research_context.answerable:

            yield {
                "type": "token",
                "text": NOT_FOUND_MESSAGE,
            }

            yield {
                "type": "sources",
                "doc_sources": [],
                "web_sources": [],
                "answerable": False,
            }

            yield {
                "type": "done"
            }

            return

        # ---------------------------------------------------------------
        # 9. Reflection on web evidence
        # ---------------------------------------------------------------

        yield {
            "type": "status",
            "message": "Checking the web research...",
        }

        web_verdict = self.reflection_agent.validate(
            search_query,
            research_context.evidence,
        )

        if not web_verdict.sufficient:

            yield {
                "type": "token",
                "text": NOT_FOUND_MESSAGE,
            }

            yield {
                "type": "sources",
                "doc_sources": [],
                "web_sources": research_context.sources,
                "answerable": False,
            }

            yield {
                "type": "done"
            }

            return

        # ---------------------------------------------------------------
        # 10. Stream final web-grounded answer
        # ---------------------------------------------------------------

        yield {
            "type": "status",
            "message": "Writing answer from web research...",
        }

        for token in self.web_research_agent.generate_stream(
            search_query,
            research_context.evidence,
        ):
            yield {
                "type": "token",
                "text": token,
            }

        yield {
            "type": "sources",
            "doc_sources": [],
            "web_sources": research_context.sources,
            "answerable": True,
        }

        yield {
            "type": "done"
        }

    # -----------------------------------------------------------------------
    # Reflection debugging
    # -----------------------------------------------------------------------

    @staticmethod
    def _debug_reflection_context(
        query: str,
        results: list[RetrievalResult],
        evidence: str,
    ) -> None:
        """
        Print exactly what is being passed from retrieval to
        ReflectionAgent.

        This is temporary diagnostic logging.

        It helps determine whether:
            Retrieval → Reflection
        is working correctly.
        """

        print("\n" + "=" * 100)

        print(
            "[REFLECTION DEBUG] QUERY"
        )

        print(
            query
        )

        print(
            "\n[REFLECTION DEBUG] "
            "RETRIEVED DOCUMENT RESULTS"
        )

        print(
            f"Results passed to ReflectionAgent: "
            f"{len(results)}"
        )

        for i, result in enumerate(
            results,
            start=1,
        ):

            print(
                "\n" + "-" * 80
            )

            print(
                f"RESULT #{i}"
            )

            print(
                f"Score: {result.score}"
            )

            print(
                f"Source: {result.source_filename}"
            )

            print(
                f"Page: "
                f"{getattr(result.chunk, 'page_number', None)}"
            )

            print(
                "\nTEXT:"
            )

            print(
                result.chunk.text
            )

        print(
            "\n" + "-" * 80
        )

        print(
            "[REFLECTION DEBUG] "
            "FORMATTED EVIDENCE LENGTH:"
        )

        print(
            len(evidence)
        )

        print(
            "\n[REFLECTION DEBUG] "
            "FORMATTED EVIDENCE:"
        )

        print(
            evidence
        )

        print(
            "\n" + "=" * 100
        )

    # -----------------------------------------------------------------------
    # Intent classification
    # -----------------------------------------------------------------------

    def _classify_intent(
        self,
        message: str,
    ) -> str:

        response = generate_with_retry(
            self.client,
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

        intent = (
            response.text or ""
        ).strip().upper()

        if intent == "EMAIL":
            return "EMAIL"

        if intent == "GREETING":
            return "GREETING"

        return "QUESTION"

    # -----------------------------------------------------------------------
    # Standalone query
    # -----------------------------------------------------------------------

    def _standalone_query(
        self,
        message: str,
        history: list[dict] | None,
    ) -> str:
        """
        Rewrites a follow-up question into a standalone question.

        If there is no history, no LLM call is made.
        """

        if not history:
            return message

        history_text = "\n".join(
            f"User: {turn['question']}\n"
            f"Assistant: {turn['answer']}"
            for turn in history[-6:]
        )

        prompt = f"""
Conversation history:
{history_text}

Follow-up question:
{message}
"""

        response = generate_with_retry(
            self.client,
            model=self.model,
            contents=prompt,
            config={
                "system_instruction": STANDALONE_QUERY_PROMPT,
                "max_output_tokens": 150,
                "thinking_config": {
                    "thinking_budget": 0
                },
            },
        )

        rewritten = (
            response.text or ""
        ).strip()

        return rewritten or message

    # -----------------------------------------------------------------------
    # Generate document answer
    # -----------------------------------------------------------------------

    def _generate(
        self,
        question: str,
        evidence: str,
    ) -> str:

        prompt = f"""
Evidence:
{evidence}

Question:
{question}
"""

        response = generate_with_retry(
            self.client,
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

    # -----------------------------------------------------------------------
    # Streaming document answer
    # -----------------------------------------------------------------------

    def _generate_stream(
        self,
        question: str,
        evidence: str,
    ):

        prompt = f"""
Evidence:
{evidence}

Question:
{question}
"""

        stream = generate_stream_with_retry(
            self.client,
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

        for chunk in stream:

            if chunk.text:
                yield chunk.text

    # -----------------------------------------------------------------------
    # Format document evidence
    # -----------------------------------------------------------------------

    @staticmethod
    def _format_doc_evidence(
        results: list[RetrievalResult],
    ) -> str:

        blocks = []

        for i, result in enumerate(
            results,
            start=1,
        ):

            page = (
                f", page {result.chunk.page_number}"
                if result.chunk.page_number is not None
                else ""
            )

            blocks.append(
                f"[Doc {i}: "
                f"{result.source_filename}"
                f"{page}]\n"
                f"{result.chunk.text}"
            )

        return "\n\n".join(
            blocks
        )

    # -----------------------------------------------------------------------
    # Legacy web evidence formatter
    #
    # Kept because WebSearchResult is still part of AgentAnswer and API
    # source handling.
    # -----------------------------------------------------------------------

    @staticmethod
    def _format_web_evidence(
        results: list[WebSearchResult],
    ) -> str:

        blocks = []

        for i, result in enumerate(
            results,
            start=1,
        ):

            blocks.append(
                f"[Web {i}: "
                f"{result.title} "
                f"({result.url})]\n"
                f"{result.snippet}"
            )

        return "\n\n".join(
            blocks
        )