"""
Document extraction using Gemini.

PDF:
    Used as the OCR fallback for scanned PDFs.
    The PDF is sent directly to Gemini as bytes.

Images:
    A single Gemini vision call handles both:
    1. Text/document transcription
    2. Ordinary image/scene description

Returns:
    list[tuple[str, int | None]]

For PDFs:
    (text, page_number)

For images:
    (text, None)
"""

from pathlib import Path
import re

from google import genai
from google.genai import types

from config import settings


# ---------------------------------------------------------------------------
# Supported image MIME types
# ---------------------------------------------------------------------------

_MIME_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class OCRNotConfiguredError(Exception):
    pass


# ---------------------------------------------------------------------------
# PDF extraction prompt
# ---------------------------------------------------------------------------

PDF_EXTRACTION_PROMPT = """
You are a high-accuracy document transcription engine.

Extract the complete readable content from this PDF.

This is DOCUMENT EXTRACTION, not question answering.

Do NOT summarize.
Do NOT explain.
Do NOT answer questions.
Do NOT invent information.

============================================================
PAGE MARKERS
============================================================

Start every page with exactly:

=== PAGE 1 ===
=== PAGE 2 ===
=== PAGE 3 ===

and so on.

The page number must correspond to the original PDF page.

============================================================
TEXT
============================================================

Transcribe all readable text.

Preserve exactly:

- names
- dates
- roll numbers
- enrollment numbers
- registration numbers
- addresses
- headings
- labels
- subject names
- marks
- totals
- grades
- percentages
- other important numbers

Do not silently omit information.

============================================================
TABLES
============================================================

Tables are extremely important.

Convert every table into clean HTML.

Example:

<table>
<thead>
<tr>
<th>Subject</th>
<th>Maximum Marks</th>
<th>Marks Obtained</th>
</tr>
</thead>
<tbody>
<tr>
<td>English</td>
<td>100</td>
<td>78</td>
</tr>
</tbody>
</table>

Preserve:

- all headers
- all rows
- all values
- column relationships
- row relationships

Do NOT flatten a table into random text.

This is especially important for:

- marksheets
- transcripts
- invoices
- financial statements
- result sheets
- tables containing numbers

============================================================
SCANNED PAGES
============================================================

If the PDF page is an image/scanned document, visually read it
and transcribe the readable content.

Do not say that the page cannot be processed simply because it
is scanned.

============================================================
UNREADABLE VALUES
============================================================

If a value genuinely cannot be read, use:

[UNREADABLE]

Never guess.

============================================================
OUTPUT
============================================================

Return ONLY the extracted document content.

Do not add:

- summaries
- explanations
- conclusions
- commentary
- "Here is the transcription"

Start directly with:

=== PAGE 1 ===
"""


# ---------------------------------------------------------------------------
# Image extraction prompt
# ---------------------------------------------------------------------------

IMAGE_EXTRACTION_PROMPT = """
Analyze this image for a searchable RAG knowledge base.

There are two cases.

============================================================
CASE 1 — TEXT / DOCUMENT IMAGE
============================================================

If the image contains meaningful readable text:

Transcribe ALL readable text.

Preserve:

- names
- dates
- numbers
- headings
- labels
- addresses
- signs
- document text
- subject names
- marks
- grades

If a table is visible, convert it to clean HTML.

Do not summarize the text.

============================================================
CASE 2 — ORDINARY PHOTOGRAPH
============================================================

If there is no meaningful document/text content, describe what
is visibly present.

Include:

- people
- objects
- buildings
- landmarks
- vehicles
- setting
- visible location clues

If a recognizable landmark or location is visible, identify it
when the visual evidence supports the identification.

Do not invent information.

============================================================
OUTPUT
============================================================

If text is present:
    return the transcription.

If it is an ordinary photograph:
    return a factual visual description.

Do not add unnecessary commentary.
"""


# ---------------------------------------------------------------------------
# Page marker parser
# ---------------------------------------------------------------------------

_PAGE_MARKER_RE = re.compile(
    r"(?im)^\s*===\s*PAGE\s+(\d+)\s*===\s*$"
)


def _split_pdf_pages(
    text: str,
) -> list[tuple[str, int]]:
    """
    Split Gemini's page-marked response into:

        [(page_text, page_number), ...]

    If Gemini does not return page markers, the entire response
    is safely treated as page 1.
    """

    if not text or not text.strip():
        return []

    matches = list(
        _PAGE_MARKER_RE.finditer(text)
    )

    if not matches:
        return [
            (
                text.strip(),
                1,
            )
        ]

    pages: list[tuple[str, int]] = []

    for index, match in enumerate(matches):

        page_number = int(
            match.group(1)
        )

        start = match.end()

        if index + 1 < len(matches):
            end = matches[index + 1].start()
        else:
            end = len(text)

        page_text = text[start:end].strip()

        if page_text:
            pages.append(
                (
                    page_text,
                    page_number,
                )
            )

    return pages


# ---------------------------------------------------------------------------
# PDF OCR using Gemini
# ---------------------------------------------------------------------------

def parse_pdf_via_ocr(
    filepath: str,
) -> list[tuple[str, int | None]]:
    """
    Extract a scanned PDF using Gemini.

    IMPORTANT:
    PyMuPDF remains responsible for native PDF extraction.

    This function is only called when the native PDF extraction
    determines that the PDF contains little/no usable text.

    The PDF is sent directly to Gemini as raw bytes.
    """

    path = Path(filepath)

    if not path.exists():
        raise FileNotFoundError(
            f"PDF file not found: {filepath}"
        )

    if path.suffix.lower() != ".pdf":
        raise ValueError(
            f"Expected PDF file, got: {path.suffix}"
        )

    if not settings.gemini_api_key:
        raise OCRNotConfiguredError(
            "GEMINI_API_KEY is not set in .env — "
            "scanned PDF OCR is unavailable."
        )

    print("\n" + "=" * 80)
    print("[OCR] Calling Gemini OCR for PDF")
    print(
        f"[OCR] File: {path}"
    )
    print(
        f"[OCR] Size: {path.stat().st_size} bytes"
    )
    print("=" * 80)

    try:

        # ---------------------------------------------------------------
        # Gemini client
        # ---------------------------------------------------------------

        client = genai.Client(
            api_key=settings.gemini_api_key
        )

        # ---------------------------------------------------------------
        # Read PDF as raw bytes
        #
        # No manual base64 encoding.
        # ---------------------------------------------------------------

        pdf_bytes = path.read_bytes()

        pdf_part = types.Part.from_bytes(
            data=pdf_bytes,
            mime_type="application/pdf",
        )

        # ---------------------------------------------------------------
        # Gemini document understanding
        # ---------------------------------------------------------------

        response = client.models.generate_content(
            model="gemini-3.1-flash-lite",
            contents=[
                pdf_part,
                PDF_EXTRACTION_PROMPT,
            ],
            config={
                "temperature": 0,
                "max_output_tokens": 12000,
            },
        )

        raw_text = (
            response.text or ""
        ).strip()

        if not raw_text:
            raise OCRNotConfiguredError(
                "Gemini returned an empty OCR response."
            )

        # ---------------------------------------------------------------
        # Split into pages
        # ---------------------------------------------------------------

        pages = _split_pdf_pages(
            raw_text
        )

        print(
            f"[ocr] Gemini OCR returned "
            f"{len(pages)} page(s) for {path}"
        )

        total_chars = 0

        for page_text, page_number in pages:

            total_chars += len(
                page_text
            )

            print(
                f"[ocr]   page {page_number}: "
                f"{len(page_text)} chars"
            )

            print(
                "  ----- Gemini OCR content preview -----"
            )

            preview = page_text[:1200]

            print(preview)

            if len(page_text) > 1200:
                print(
                    "  ... [truncated]"
                )

            print(
                "  --------------------------------"
            )

        print(
            f"[ocr] Gemini OCR succeeded for "
            f"{path}: {len(pages)} pages, "
            f"{total_chars} total chars"
        )

        return pages

    except Exception as exc:

        print(
            f"[ocr] Gemini OCR failed for {path}: "
            f"{type(exc).__name__}: {exc}"
        )

        raise


# ---------------------------------------------------------------------------
# Image processing using Gemini
# ---------------------------------------------------------------------------

def parse_image(
    filepath: str,
) -> list[tuple[str, int | None]]:
    """
    Process an image with Gemini.

    One Gemini call handles both:

    - OCR/transcription of text
    - description of ordinary photographs
    """

    path = Path(filepath)

    if not path.exists():
        raise FileNotFoundError(
            f"Image file not found: {filepath}"
        )

    mime_type = _MIME_TYPES.get(
        path.suffix.lower()
    )

    if mime_type is None:
        raise ValueError(
            f"Unsupported image type: {path.suffix}"
        )

    if not settings.gemini_api_key:
        raise OCRNotConfiguredError(
            "GEMINI_API_KEY is not set in .env — "
            "image processing is unavailable."
        )

    print("\n" + "=" * 80)
    print("[VISION] Calling Gemini for image")
    print(
        f"[VISION] File: {path}"
    )
    print("=" * 80)

    try:

        client = genai.Client(
            api_key=settings.gemini_api_key
        )

        # ---------------------------------------------------------------
        # Raw image bytes
        # ---------------------------------------------------------------

        image_bytes = path.read_bytes()

        image_part = types.Part.from_bytes(
            data=image_bytes,
            mime_type=mime_type,
        )

        # ---------------------------------------------------------------
        # One Gemini call
        # ---------------------------------------------------------------

        response = client.models.generate_content(
            model="gemini-3.1-flash-lite",
            contents=[
                image_part,
                IMAGE_EXTRACTION_PROMPT,
            ],
            config={
                "temperature": 0,
                "max_output_tokens": 4000,
            },
        )

        text = (
            response.text or ""
        ).strip()

        if not text:
            print(
                "[VISION] Gemini returned no content."
            )
            return []

        print(
            f"[VISION] Gemini returned "
            f"{len(text)} characters"
        )

        print(
            "----- Gemini image result -----"
        )

        print(
            text[:2000]
        )

        if len(text) > 2000:
            print(
                "... [truncated]"
            )

        print(
            "--------------------------------"
        )

        return [
            (
                text,
                None,
            )
        ]

    except Exception as exc:

        print(
            f"[VISION] Gemini image processing failed: "
            f"{type(exc).__name__}: {exc}"
        )

        raise