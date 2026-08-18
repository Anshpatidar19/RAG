"""
Email Writing Agent.

Generates professional email drafts from the user's instructions.

This agent does NOT send emails.
It only creates the draft.
"""

from google import genai

from agents.retry_utils import generate_with_retry


SYSTEM_PROMPT = """
You are a professional email writing assistant.

Your job is to turn the user's request into a clear, natural,
professional email.

Rules:

1. Generate an appropriate subject line.
2. Use a professional tone unless the user requests another tone.
3. Do not invent facts, dates, names, reasons, or commitments.
4. If important information is missing, use a clear placeholder such as:
   [Recipient Name]
   [Date]
   [Reason]
5. Keep the email concise unless the user asks for detail.
6. Do not add explanations before or after the email.
7. Return exactly this structure:

Subject: <subject>

<email body>
"""


class EmailWriterAgent:
    def __init__(
        self,
        client: genai.Client,
        model: str = "gemini-3.1-flash-lite",
    ):
        self.client = client
        self.model = model

    def write(self, request: str) -> str:
        response = generate_with_retry(
            self.client,
            model=self.model,
            contents=request,
            config={
                "system_instruction": SYSTEM_PROMPT,
                "max_output_tokens": 700,
                "thinking_config": {"thinking_budget": 0},
            },
        )

        return (response.text or "").strip()

    def write_stream(self, request: str):
        """
        Streaming version for /ask/stream.
        """

        from agents.retry_utils import generate_stream_with_retry

        stream = generate_stream_with_retry(
            self.client,
            model=self.model,
            contents=request,
            config={
                "system_instruction": SYSTEM_PROMPT,
                "max_output_tokens": 700,
                "thinking_config": {"thinking_budget": 0},
            },
        )

        for chunk in stream:
            if chunk.text:
                yield chunk.text