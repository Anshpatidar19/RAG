"""
Parses raw files into plain text. Each parser returns a list of
(text, page_number) tuples so the chunker can preserve page-level
metadata for source attribution. page_number is None for formats
that don't have pages (TXT, MD).
"""

from pathlib import Path

import fitz  # PyMuPDF
from docx import Document as DocxDocument

from ingestion.ocr_parser import parse_image


class UnsupportedFileTypeError(Exception):
    pass


def parse_pdf(filepath: str) -> list[tuple[str, int | None]]:
    """Returns one (text, page_number) tuple per page."""
    pages = []
    with fitz.open(filepath) as doc:
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text()
            if text.strip():
                pages.append((text, page_num))
    return pages


def parse_docx(filepath: str) -> list[tuple[str, int | None]]:
    """DOCX has no native page concept, so everything is one block."""
    doc = DocxDocument(filepath)
    full_text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    return [(full_text, None)] if full_text.strip() else []


def parse_txt(filepath: str) -> list[tuple[str, int | None]]:
    text = Path(filepath).read_text(encoding="utf-8", errors="ignore")
    return [(text, None)] if text.strip() else []


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


def parse_document(filepath: str) -> list[tuple[str, int | None]]:
    """Dispatches to the right parser based on file extension."""
    ext = Path(filepath).suffix.lower()
    parser = PARSERS.get(ext)
    if parser is None:
        raise UnsupportedFileTypeError(
            f"No parser for '{ext}' files. Supported: {list(PARSERS.keys())}"
        )
    return parser(filepath)