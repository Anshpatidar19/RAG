"""
Document chunking.

For OCR/document pages, we keep each page together as one chunk whenever
possible. This preserves the relationship between:

    document heading
    student/person name
    semester
    subjects
    marks
    total/result

This is especially important for scanned marksheets and table-heavy PDFs.

If a page is larger than the configured chunk size, the page is split
using RecursiveCharacterTextSplitter while preserving the page_number.
"""

import uuid

from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import settings
from core.models import Chunk


def chunk_document(
    doc_id: str,
    pages: list[tuple[str, int | None]],
) -> list[Chunk]:
    """
    Convert parsed pages into Chunks.

    Strategy:

    1. Keep each page as ONE chunk if it fits inside chunk_size.
    2. If a page is larger than chunk_size, split that page using
       RecursiveCharacterTextSplitter.
    3. Preserve page_number for every generated chunk.
    4. Keep chunk_index sequential across the whole document.

    This prevents important information from a single OCR page
    (for example, the student's name and GRAND TOTAL) from being
    separated unnecessarily.
    """

    # ---------------------------------------------------------------
    # Fallback splitter for pages that are genuinely too large.
    # ---------------------------------------------------------------

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
    )

    chunks: list[Chunk] = []

    chunk_index = 0

    # ---------------------------------------------------------------
    # Process each page independently.
    # ---------------------------------------------------------------

    for text, page_number in pages:

        if not text or not text.strip():
            continue

        text = text.strip()

        # ===========================================================
        # IMPORTANT:
        #
        # Keep the COMPLETE page together when it fits.
        # ===========================================================

        if len(text) <= settings.chunk_size:

            chunks.append(
                Chunk(
                    chunk_id=str(uuid.uuid4()),
                    doc_id=doc_id,
                    text=text,
                    chunk_index=chunk_index,
                    page_number=page_number,
                )
            )

            chunk_index += 1

            continue

        # ===========================================================
        # Page is larger than chunk_size.
        #
        # Split only this page.
        # ===========================================================

        splits = splitter.split_text(text)

        for split_text in splits:

            if not split_text.strip():
                continue

            chunks.append(
                Chunk(
                    chunk_id=str(uuid.uuid4()),
                    doc_id=doc_id,
                    text=split_text.strip(),
                    chunk_index=chunk_index,
                    page_number=page_number,
                )
            )

            chunk_index += 1

    # ---------------------------------------------------------------
    # Debug information
    # ---------------------------------------------------------------

    print(
        "\n" + "=" * 80
    )

    print(
        "[CHUNKING] Document chunking complete"
    )

    print(
        f"[CHUNKING] Document ID: {doc_id}"
    )

    print(
        f"[CHUNKING] Pages received: {len(pages)}"
    )

    print(
        f"[CHUNKING] Chunks created: {len(chunks)}"
    )

    print(
        f"[CHUNKING] Configured chunk size: "
        f"{settings.chunk_size}"
    )

    print(
        f"[CHUNKING] Configured overlap: "
        f"{settings.chunk_overlap}"
    )

    # ---------------------------------------------------------------
    # Print page/chunk mapping
    # ---------------------------------------------------------------

    for chunk in chunks:

        print(
            f"[CHUNKING] "
            f"chunk_index={chunk.chunk_index} "
            f"page={chunk.page_number} "
            f"chars={len(chunk.text)}"
        )

    print(
        "=" * 80
    )

    return chunks