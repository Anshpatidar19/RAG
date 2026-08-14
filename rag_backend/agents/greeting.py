"""
Handles chit-chat / meta messages ("hi", "who are you", "thanks") directly,
with no retrieval involved. This is what prevents the system from
hallucinating a document-grounded answer to a message that was never
a real question in the first place.
"""

from google import genai

SYSTEM_PROMPT = """You are a friendly assistant for a document Q&A system. \
The user has sent a casual/conversational message (greeting, thanks, small talk, \
a question about you/your capabilities) rather than a question about their documents.

Respond briefly and naturally. If relevant, mention you can answer questions about \
their uploaded documents. Do not fabricate any information about specific documents \
— you have not been given any document content for this message."""


class GreetingAgent:
    def __init__(self, client: genai.Client, model: str = "gemini-3.1-flash-lite"):
        self.client = client
        self.model = model

    def respond(self, message: str) -> str:
        response = self.client.models.generate_content(
            model=self.model,
            contents=message,
            config={
                "system_instruction": SYSTEM_PROMPT,
                "max_output_tokens": 256,
                "thinking_config": {"thinking_budget": 0},
            },
        )
        return response.text or ""

    def respond_stream(self, message: str):
        stream = self.client.models.generate_content_stream(
            model=self.model,
            contents=message,
            config={
                "system_instruction": SYSTEM_PROMPT,
                "max_output_tokens": 256,
                "thinking_config": {"thinking_budget": 0},
            },
        )
        for chunk in stream:
            if chunk.text:
                yield chunk.text