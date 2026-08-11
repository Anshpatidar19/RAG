"""
Repository layer over Pinecone. All add/query/delete/update operations
against the vector index go through this class — nothing else in the
codebase should import pinecone directly.
"""

from pinecone import Pinecone

from config import settings
from core.models import Chunk, RetrievalResult


class VectorStoreRepository:
    def __init__(self):
        self._client = Pinecone(api_key=settings.pinecone_api_key)
        self._index = self._client.Index(settings.pinecone_index_name)

    def upsert_chunks(
        self, chunks: list[Chunk], vectors: list[list[float]], source_filename: str
    ) -> None:
        """Add or overwrite chunks in the index."""
        records = []
        for chunk, vector in zip(chunks, vectors):
            records.append(
                {
                    "id": chunk.chunk_id,
                    "values": vector,
                    "metadata": {
                        "doc_id": chunk.doc_id,
                        "text": chunk.text,
                        "chunk_index": chunk.chunk_index,
                        "page_number": chunk.page_number
                        if chunk.page_number is not None
                        else -1,
                        "source_filename": source_filename,
                    },
                }
            )
        self._index.upsert(vectors=records)

    def query(self, query_vector: list[float], top_k: int) -> list[RetrievalResult]:
        response = self._index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True,
        )
        results = []
        for match in response.matches:
            meta = match.metadata
            chunk = Chunk(
                chunk_id=match.id,
                doc_id=meta["doc_id"],
                text=meta["text"],
                chunk_index=meta["chunk_index"],
                page_number=meta["page_number"] if meta["page_number"] != -1 else None,
            )
            results.append(
                RetrievalResult(
                    chunk=chunk,
                    score=match.score,
                    source_filename=meta["source_filename"],
                )
            )
        return results

    def delete_by_doc_id(self, doc_id: str) -> None:
        """Delete all chunks belonging to a document."""
        self._index.delete(filter={"doc_id": {"$eq": doc_id}})

    def delete_all(self) -> None:
        """Wipe the entire index. Use with care."""
        self._index.delete(delete_all=True)