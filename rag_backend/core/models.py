"""
Core data structures shared across the RAG system.
"""

from dataclasses import dataclass, field
from datetime import datetime

from core.enums import DocumentStatus


@dataclass
class Document:
    doc_id: str
    filename: str
    filepath: str
    status: DocumentStatus
    uploaded_at: datetime
    num_chunks: int = 0
    content_hash: str | None = None


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    chunk_index: int
    page_number: int | None = None


@dataclass
class RetrievalResult:
    chunk: Chunk
    score: float
    source_filename: str


@dataclass
class Answer:
    text: str
    sources: list[RetrievalResult] = field(default_factory=list)
    answerable: bool = True