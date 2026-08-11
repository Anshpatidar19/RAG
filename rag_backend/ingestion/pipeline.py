"""
Orchestrates the full ingestion flow: parse -> chunk -> embed -> store.
This is what add_document/update_document call into from the KB facade.
"""

import uuid
from datetime import datetime
from pathlib import Path

from core.enums import DocumentStatus
from core.models import Document
from embedding.embedder import Embedder
from ingestion.chunker import chunk_document
from ingestion.parsers import parse_document
from storage.vector_store import VectorStoreRepository


class IngestionPipeline:
    def __init__(self, embedder: Embedder, vector_store: VectorStoreRepository):
        self.embedder = embedder
        self.vector_store = vector_store

    def ingest(
        self, filepath: str, doc_id: str | None = None, display_filename: str | None = None
    ) -> Document:
        """
        Parses, chunks, embeds, and stores a document.
        Pass an existing doc_id to re-ingest under the same id (used by update()).
        Otherwise a new doc_id is generated.
        display_filename overrides the name derived from filepath — use this
        when filepath is a temp file and you want the original upload name
        stored as metadata instead.
        """
        doc_id = doc_id or str(uuid.uuid4())
        filename = display_filename or Path(filepath).name

        document = Document(
            doc_id=doc_id,
            filename=filename,
            filepath=filepath,
            status=DocumentStatus.PROCESSING,
            uploaded_at=datetime.utcnow(),
        )

        try:
            pages = parse_document(filepath)
            chunks = chunk_document(doc_id, pages)

            if not chunks:
                document.status = DocumentStatus.FAILED
                return document

            texts = [chunk.text for chunk in chunks]
            vectors = self.embedder.embed_batch(texts)

            self.vector_store.upsert_chunks(
                chunks=chunks, vectors=vectors, source_filename=filename
            )

            document.status = DocumentStatus.INDEXED
            document.num_chunks = len(chunks)
            return document

        except Exception:
            document.status = DocumentStatus.FAILED
            raise

    def update(
        self, doc_id: str, filepath: str, display_filename: str | None = None
    ) -> Document:
        """Deletes the old version's chunks, then re-ingests under the same doc_id."""
        self.vector_store.delete_by_doc_id(doc_id)
        return self.ingest(filepath, doc_id=doc_id, display_filename=display_filename)