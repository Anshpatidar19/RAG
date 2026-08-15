"""
Persists Document metadata in Supabase (Postgres) instead of local SQLite,
so it survives across machines/restarts and supports dedup lookups by
content hash. The actual chunk content and vectors still live in Pinecone —
this table only tracks bookkeeping (filename, status, chunk count, hash).
"""

from datetime import datetime

from supabase import Client, create_client

from config import settings
from core.enums import DocumentStatus
from core.models import Document


class DocumentRepository:
    def __init__(self, client: Client | None = None):
        self._client = client or create_client(settings.supabase_url, settings.supabase_key)

    def save(self, document: Document) -> None:
        """Insert or replace — used for both add and update."""
        self._client.table("documents").upsert(
            {
                "doc_id": document.doc_id,
                "filename": document.filename,
                "filepath": document.filepath,
                "status": document.status.value,
                "uploaded_at": document.uploaded_at.isoformat(),
                "num_chunks": document.num_chunks,
                "content_hash": document.content_hash,
            }
        ).execute()

    def get(self, doc_id: str) -> Document | None:
        response = (
            self._client.table("documents").select("*").eq("doc_id", doc_id).execute()
        )
        rows = response.data
        return self._row_to_document(rows[0]) if rows else None

    def get_by_content_hash(self, content_hash: str) -> Document | None:
        response = (
            self._client.table("documents")
            .select("*")
            .eq("content_hash", content_hash)
            .execute()
        )
        rows = response.data
        return self._row_to_document(rows[0]) if rows else None

    def list_all(self) -> list[Document]:
        response = (
            self._client.table("documents")
            .select("*")
            .order("uploaded_at", desc=True)
            .execute()
        )
        return [self._row_to_document(row) for row in response.data]

    def delete(self, doc_id: str) -> None:
        self._client.table("documents").delete().eq("doc_id", doc_id).execute()

    @staticmethod
    def _row_to_document(row: dict) -> Document:
        return Document(
            doc_id=row["doc_id"],
            filename=row["filename"],
            filepath=row["filepath"],
            status=DocumentStatus(row["status"]),
            uploaded_at=datetime.fromisoformat(row["uploaded_at"]),
            num_chunks=row["num_chunks"],
            content_hash=row.get("content_hash"),
        )