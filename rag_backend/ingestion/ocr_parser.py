"""
Parses image files (screenshots, photographed pages, diagrams with text)
into text using Mistral's OCR API. Output flows into the same
chunk -> embed -> store pipeline as PDF/DOCX/TXT parsers.
"""

import base64
from pathlib import Path

from mistralai import Mistral

from config import settings

_MIME_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}


class OCRNotConfiguredError(Exception):
    pass


def parse_image(filepath: str) -> list[tuple[str, int | None]]:
    if not settings.mistral_api_key:
        raise OCRNotConfiguredError(
            "MISTRAL_API_KEY is not set in .env — image OCR is unavailable."
        )

    path = Path(filepath)
    mime_type = _MIME_TYPES.get(path.suffix.lower(), "image/png")
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")

    client = Mistral(api_key=settings.mistral_api_key)
    response = client.ocr.process(
        model="mistral-ocr-latest",
        document={
            "type": "image_url",
            "image_url": f"data:{mime_type};base64,{encoded}",
        },
    )

    pages: list[tuple[str, int | None]] = []
    for page in response.pages:
        if page.markdown and page.markdown.strip():
            pages.append((page.markdown, None))
    return pages