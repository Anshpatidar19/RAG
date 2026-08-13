"""
Evidence Validator. Judges whether a set of retrieved results (from vector
search or web search) actually contains enough to answer the question —
this is the main defense against hallucination beyond the raw similarity
score threshold. A chunk can be the "closest match" without actually
answering the question; this agent checks for that gap.
"""

from dataclasses import dataclass

from google import genai

SYSTEM_PROMPT = """You judge whether the given evidence is sufficient to answer \
the given question.

Respond with EXACTLY one word on the first line:
- SUFFICIENT — the evidence directly answers the question
- INSUFFICIENT — the evidence is missing, off-topic, or only tangentially related

On the second line, give a one-sentence reason."""


@dataclass
class EvidenceVerdict:
    sufficient: bool
    reason: str


class ReflectionAgent:
    def __init__(self, client: genai.Client, model: str = "gemini-3.1-flash-lite"):
        self.client = client
        self.model = model

    def validate(self, question: str, evidence_text: str) -> EvidenceVerdict:
        if not evidence_text.strip():
            return EvidenceVerdict(sufficient=False, reason="No evidence retrieved.")

        prompt = f"""Question: {question}

Evidence:
{evidence_text}"""

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config={
                "system_instruction": SYSTEM_PROMPT,
                "max_output_tokens": 150,
                "thinking_config": {"thinking_budget": 0},
            },
        )

        text = response.text or ""
        lines = text.strip().splitlines()
        verdict_line = lines[0].strip().upper() if lines else ""
        reason = lines[1].strip() if len(lines) > 1 else ""

        return EvidenceVerdict(sufficient="SUFFICIENT" in verdict_line, reason=reason)