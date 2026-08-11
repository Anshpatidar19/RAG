"""
Persists Document metadata (filename, status, chunk count) in SQLite so
the document list survives server restarts. The actual chunk content
and vectors live in Pinecone — this table only tracks bookkeeping.
"""

import sqlite3
from datetime import datetime
from pathlib import Path

from core.enums import DocumentStatus
from core.models import Document

DB_PATH = Path(__file__).parent.parent / "rag_metadata.db"


class DocumentRepository:
    def __init__(self, db_path: Path | str = DB_PATH):
        self.db_path = str(db_path)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    filepath TEXT NOT NULL,
                    status TEXT NOT NULL,
                    uploaded_at TEXT NOT NULL,
                    num_chunks INTEGER NOT NULL DEFAULT 0
                )
                """
            )

    def save(self, document: Document) -> None:
        """Insert or replace — used for both add and update."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO documents (doc_id, filename, filepath, status, uploaded_at, num_chunks)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(doc_id) DO UPDATE SET
                    filename=excluded.filename,
                    filepath=excluded.filepath,
                    status=excluded.status,
                    uploaded_at=excluded.uploaded_at,
                    num_chunks=excluded.num_chunks
                """,
                (
                    document.doc_id,
                    document.filename,
                    document.filepath,
                    document.status.value,
                    document.uploaded_at.isoformat(),
                    document.num_chunks,
                ),
            )

    def get(self, doc_id: str) -> Document | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE doc_id = ?", (doc_id,)
            ).fetchone()
            return self._row_to_document(row) if row else None

    def list_all(self) -> list[Document]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM documents ORDER BY uploaded_at DESC"
            ).fetchall()
            return [self._row_to_document(row) for row in rows]

    def delete(self, doc_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))

    @staticmethod
    def _row_to_document(row: sqlite3.Row) -> Document:
        return Document(
            doc_id=row["doc_id"],
            filename=row["filename"],
            filepath=row["filepath"],
            status=DocumentStatus(row["status"]),
            uploaded_at=datetime.fromisoformat(row["uploaded_at"]),
            num_chunks=row["num_chunks"],
        )