"""
Splits parsed text into overlapping chunks, producing Chunk objects
ready to be embedded and stored. Uses langchain's RecursiveCharacterTextSplitter
so splits happen on natural boundaries (paragraphs, then sentences,
then words) rather than mid-word.
"""

import uuid

from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import settings
from core.models import Chunk


def chunk_document(
    doc_id: str, pages: list[tuple[str, int | None]]
) -> list[Chunk]:
    """
    Takes the (text, page_number) tuples from a parser and returns
    a flat list of Chunks with sequential chunk_index and, where
    available, the originating page_number preserved.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[Chunk] = []
    chunk_index = 0

    for text, page_number in pages:
        splits = splitter.split_text(text)
        for split_text in splits:
            if not split_text.strip():
                continue
            chunks.append(
                Chunk(
                    chunk_id=str(uuid.uuid4()),
                    doc_id=doc_id,
                    text=split_text,
                    chunk_index=chunk_index,
                    page_number=page_number,
                )
            )
            chunk_index += 1

    return chunks