"""
Parses image files into text for the ingestion pipeline. Two distinct
capabilities are combined here:

1. OCR (Mistral) — extracts text that's actually WRITTEN in the image
   (scanned documents, screenshots, signs, marksheets). This is the
   primary path.
2. Vision captioning (Gemini) — used only as a fallback when OCR finds
   no readable text at all, e.g. an ordinary photo (a person standing
   in front of a landmark). OCR can't identify scenery, objects, or
   locations — that needs an actual vision model to describe what's
   IN the photo, not what's written on it.
"""

import base64
from pathlib import Path

from google import genai
from mistralai import Mistral

from config import settings

_MIME_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}

_CAPTION_PROMPT = (
    "Describe this image factually and specifically. Include any people, "
    "objects, setting, and — importantly — any recognizable landmarks, "
    "buildings, or locations visible in the photo. If you recognize the "
    "location (e.g. a famous monument), name it explicitly."
)


class OCRNotConfiguredError(Exception):
    pass


def _caption_with_vision(filepath: str) -> str:
    """Fallback for photos with no readable text — describes what's IN the image."""
    path = Path(filepath)
    mime_type = _MIME_TYPES.get(path.suffix.lower(), "image/png")
    client = genai.Client(api_key=settings.gemini_api_key)

    response = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=[
            {"inline_data": {"mime_type": mime_type, "data": path.read_bytes()}},
            _CAPTION_PROMPT,
        ],
        config={"max_output_tokens": 400, "thinking_config": {"thinking_budget": 0}},
    )
    return response.text or ""


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

    if not pages:
        # No text found at all — likely an ordinary photo, not a document.
        # Fall back to vision captioning so it's still searchable
        # (e.g. "a person in front of the Taj Mahal" becomes findable text).
        caption = _caption_with_vision(filepath)
        if caption.strip():
            pages.append((caption, None))

    return pages


def parse_pdf_via_ocr(filepath: str) -> list[tuple[str, int | None]]:
    """
    Used as a fallback when a PDF has little or no extractable text —
    typically a scanned document (photographed or scanned pages saved
    as a PDF with no underlying text layer). Preserves page numbers,
    unlike the image OCR path, since Mistral returns per-page results.
    """
    if not settings.mistral_api_key:
        raise OCRNotConfiguredError(
            "MISTRAL_API_KEY is not set in .env — scanned PDF OCR is unavailable."
        )

    path = Path(filepath)
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")

    client = Mistral(api_key=settings.mistral_api_key)
    response = client.ocr.process(
        model="mistral-ocr-latest",
        document={
            "type": "document_url",
            "document_url": f"data:application/pdf;base64,{encoded}",
        },
    )

    pages: list[tuple[str, int | None]] = []
    for i, page in enumerate(response.pages, start=1):
        if page.markdown and page.markdown.strip():
            pages.append((page.markdown, i))
    return pages