"""
Calls Gemini with the retrieved chunks injected as context, and returns
an Answer with the sources attached. The prompt strictly instructs the
model to only answer from the given context (point 10: say when info
isn't available) as a second layer on top of the retriever's score
threshold check.
"""

from google import genai

from config import settings
from core.models import Answer, RetrievalResult

NOT_FOUND_MESSAGE = "I don't have information about that in the knowledge base."

SYSTEM_PROMPT = """You are a helpful assistant that answers questions using ONLY \
the provided context from the user's documents.

Rules:
- Only use information present in the context below. Do not use outside knowledge.
- If the context does not contain enough information to answer the question, \
respond with exactly: "I don't have information about that in the knowledge base."
- Be concise and direct.
- Do not mention "the context" or "the documents" in your answer — just answer \
naturally, as if you know this information."""


class AnswerGenerator:
    def __init__(self, client: genai.Client | None = None, model: str = "gemini-3.5-flash"):
        self.client = client or genai.Client(api_key=settings.gemini_api_key)
        self.model = model

    def generate(self, query: str, results: list[RetrievalResult]) -> Answer:
        if not results:
            return Answer(text=NOT_FOUND_MESSAGE, sources=[], answerable=False)

        context_block = self._build_context_block(results)

        user_message = f"""Context:
{context_block}

Question: {query}"""

        response = self.client.models.generate_content(
            model=self.model,
            contents=user_message,
            config={
                "system_instruction": SYSTEM_PROMPT,
                "max_output_tokens": 1024,
            },
        )

        answer_text = response.text
        answerable = NOT_FOUND_MESSAGE not in answer_text

        return Answer(
            text=answer_text,
            sources=results if answerable else [],
            answerable=answerable,
        )

    @staticmethod
    def _build_context_block(results: list[RetrievalResult]) -> str:
        blocks = []
        for i, result in enumerate(results, start=1):
            page_info = (
                f", page {result.chunk.page_number}"
                if result.chunk.page_number is not None
                else ""
            )
            blocks.append(
                f"[Source {i}: {result.source_filename}{page_info}]\n{result.chunk.text}"
            )
        return "\n\n".join(blocks)