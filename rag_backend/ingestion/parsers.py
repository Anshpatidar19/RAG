"""
Parses raw files into plain text.

Each parser returns a list of:

    (text, page_number)

Page numbers are preserved for PDFs.
For formats without a reliable page concept, page_number is None.

PDF:
    PyMuPDF native extraction
        ↓
    Gemini OCR fallback for scanned PDFs

Images:
    Gemini vision
"""

from pathlib import Path

import fitz  # PyMuPDF

from docx import Document as DocxDocument

from ingestion.ocr_parser import (
    parse_image,
    parse_pdf_via_ocr,
)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class UnsupportedFileTypeError(Exception):
    pass


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------

def parse_pdf(
    filepath: str,
) -> list[tuple[str, int | None]]:
    """
    Extract text from a PDF.

    First:
        PyMuPDF native text extraction.

    If the PDF contains little/no text:
        Gemini document OCR.

    This preserves the original architecture while replacing
    only the Mistral OCR component.
    """

    pages: list[
        tuple[str, int | None]
    ] = []

    with fitz.open(filepath) as doc:

        for page_num, page in enumerate(
            doc,
            start=1,
        ):

            text = page.get_text()

            if text.strip():

                pages.append(
                    (
                        text,
                        page_num,
                    )
                )

    total_chars = sum(
        len(text.strip())
        for text, _ in pages
    )

    total_pdf_pages = 0

    with fitz.open(filepath) as doc:
        total_pdf_pages = len(doc)

    looks_scanned = (
        total_chars < 100
        or len(pages) < max(
            1,
            total_pdf_pages // 2,
        )
    )

    print(
        f"[ingestion] PDF native extraction: "
        f"{filepath} | "
        f"{len(pages)}/{total_pdf_pages} pages had text | "
        f"{total_chars} total chars | "
        f"looks_scanned={looks_scanned}"
    )

    # -----------------------------------------------------------------------
    # Gemini OCR fallback
    # -----------------------------------------------------------------------

    if looks_scanned:

        print(
            f"[ingestion] Falling back to Gemini OCR for "
            f"{filepath}"
        )

        ocr_pages = parse_pdf_via_ocr(
            filepath
        )

        if ocr_pages:

            print(
                f"[ingestion] Gemini OCR succeeded for "
                f"{filepath}: "
                f"{len(ocr_pages)} pages"
            )

            return ocr_pages

        print(
            f"[ingestion] Gemini OCR returned no content "
            f"for {filepath}"
        )

    # -----------------------------------------------------------------------
    # Return native extraction
    # -----------------------------------------------------------------------

    return pages


# ---------------------------------------------------------------------------
# DOCX
# ---------------------------------------------------------------------------

def parse_docx(
    filepath: str,
) -> list[tuple[str, int | None]]:
    """
    Extract text from DOCX.
    """

    doc = DocxDocument(
        filepath
    )

    text = "\n".join(
        paragraph.text
        for paragraph in doc.paragraphs
        if paragraph.text.strip()
    )

    if not text.strip():
        return []

    return [
        (
            text,
            None,
        )
    ]


# ---------------------------------------------------------------------------
# TXT / Markdown
# ---------------------------------------------------------------------------

def parse_txt(
    filepath: str,
) -> list[tuple[str, int | None]]:
    """
    Extract text from TXT or Markdown.
    """

    text = Path(filepath).read_text(
        encoding="utf-8",
        errors="ignore",
    )

    if not text.strip():
        return []

    return [
        (
            text,
            None,
        )
    ]


# ---------------------------------------------------------------------------
# Parser registry
# ---------------------------------------------------------------------------

PARSERS = {

    ".pdf": parse_pdf,

    ".docx": parse_docx,

    ".txt": parse_txt,
    ".md": parse_txt,

    ".png": parse_image,
    ".jpg": parse_image,
    ".jpeg": parse_image,
    ".webp": parse_image,
}


# ---------------------------------------------------------------------------
# Main dispatcher
# ---------------------------------------------------------------------------

def parse_document(
    filepath: str,
) -> list[tuple[str, int | None]]:

    ext = Path(filepath).suffix.lower()

    parser = PARSERS.get(
        ext
    )

    if parser is None:

        raise UnsupportedFileTypeError(
            f"No parser available for '{ext}'. "
            f"Supported file types: "
            f"{list(PARSERS.keys())}"
        )

    return parser(
        filepath
    )