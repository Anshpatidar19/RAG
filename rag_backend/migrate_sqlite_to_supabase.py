"""
One-time migration: copies document metadata from the old local
rag_metadata.db (SQLite) into Supabase, so previously uploaded
documents don't need to be re-uploaded.

Run from rag_backend/, with the venv active:
    python migrate_sqlite_to_supabase.py

Safe to run more than once — uses upsert, so re-running just overwrites
the same rows rather than duplicating them.
"""

import hashlib
import sqlite3
from datetime import datetime
from pathlib import Path

from core.enums import DocumentStatus
from core.models import Document
from storage.document_repository import DocumentRepository

SQLITE_PATH = Path(__file__).parent / "rag_metadata.db"
FILES_DIR = Path(__file__).parent / "uploaded_files"


def hash_file(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def find_file_for_doc(doc_id: str) -> Path | None:
    matches = list(FILES_DIR.glob(f"{doc_id}.*"))
    return matches[0] if matches else None


def main():
    if not SQLITE_PATH.exists():
        print(f"No local database found at {SQLITE_PATH} — nothing to migrate.")
        return

    conn = sqlite3.connect(str(SQLITE_PATH))
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM documents").fetchall()
    conn.close()

    if not rows:
        print("rag_metadata.db has no documents — nothing to migrate.")
        return

    print(f"Found {len(rows)} document(s) in the local database.")

    repo = DocumentRepository()
    migrated = 0
    skipped_hash = 0

    for row in rows:
        doc_id = row["doc_id"]
        old_filepath = row["filepath"]

        # If the original file is a URL (from a scraped page), keep it as-is.
        # Otherwise try to find the actual file on disk to compute a hash
        # for future dedup — files from before the permanent-storage change
        # may no longer exist, which is fine, they just won't dedup.
        is_url = old_filepath.startswith("http://") or old_filepath.startswith("https://")
        content_hash = None
        filepath = old_filepath

        if not is_url:
            found = find_file_for_doc(doc_id)
            if found:
                content_hash = hash_file(found)
                filepath = str(found)
            else:
                skipped_hash += 1

        document = Document(
            doc_id=doc_id,
            filename=row["filename"],
            filepath=filepath,
            status=DocumentStatus(row["status"]),
            uploaded_at=datetime.fromisoformat(row["uploaded_at"]),
            num_chunks=row["num_chunks"],
            content_hash=content_hash,
        )

        try:
            repo.save(document)
            migrated += 1
            print(f"  migrated: {document.filename}")
        except Exception as exc:
            print(f"  FAILED: {document.filename} — {exc}")

    print(f"\nDone. Migrated {migrated}/{len(rows)} document(s).")
    if skipped_hash:
        print(
            f"{skipped_hash} document(s) had no file on disk to hash — "
            f"they'll still work, but won't be deduplicated if re-uploaded."
        )


if __name__ == "__main__":
    main()