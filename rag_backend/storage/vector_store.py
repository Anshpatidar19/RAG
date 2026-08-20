"""
Repository layer over Pinecone.

All Pinecone operations go through this class.

Supports:
    - upsert
    - semantic search
    - page-level expansion
    - document deletion
    - complete index deletion

Page expansion is completely generic. It does not know anything about
marksheets, invoices, resumes, contracts, etc.
"""

import time

from pinecone import Pinecone

from config import settings
from core.models import Chunk, RetrievalResult


def _with_retry(
    func,
    max_attempts: int = 3,
    base_delay: float = 0.5,
):
    """
    Retry transient Pinecone failures a small number of times.
    """

    last_exc = None

    for attempt in range(max_attempts):

        try:
            return func()

        except Exception as exc:

            last_exc = exc

            if attempt < max_attempts - 1:
                time.sleep(
                    base_delay * (2 ** attempt)
                )

    raise last_exc


class VectorStoreRepository:

    def __init__(self):

        self._client = Pinecone(
            api_key=settings.pinecone_api_key
        )

        self._index = self._client.Index(
            settings.pinecone_index_name
        )

    # ==================================================================
    # UPSERT
    # ==================================================================

    def upsert_chunks(
        self,
        chunks: list[Chunk],
        vectors: list[list[float]],
        source_filename: str,
    ) -> None:
        """
        Add or overwrite chunks in Pinecone.
        """

        records = []

        for chunk, vector in zip(
            chunks,
            vectors,
        ):

            records.append(
                {
                    "id": chunk.chunk_id,

                    "values": vector,

                    "metadata": {
                        "doc_id": chunk.doc_id,
                        "text": chunk.text,
                        "chunk_index": chunk.chunk_index,

                        "page_number": (
                            chunk.page_number
                            if chunk.page_number is not None
                            else -1
                        ),

                        "source_filename": source_filename,
                    },
                }
            )

        _with_retry(
            lambda: self._index.upsert(
                vectors=records
            )
        )

    # ==================================================================
    # NORMAL VECTOR SEARCH
    # ==================================================================

    def query(
        self,
        query_vector: list[float],
        top_k: int,
    ) -> list[RetrievalResult]:
        """
        Semantic vector search.
        """

        response = _with_retry(
            lambda: self._index.query(
                vector=query_vector,
                top_k=top_k,
                include_metadata=True,
            )
        )

        return self._matches_to_results(
            response.matches
        )

    # ==================================================================
    # PAGE EXPANSION
    # ==================================================================

    def get_page_chunks(
        self,
        doc_id: str,
        page_number: int | float,
        query_vector: list[float],
        top_k: int = 100,
    ) -> list[RetrievalResult]:
        """
        Retrieve all chunks belonging to a particular document page.

        This uses Pinecone metadata filtering.

        It is completely document-agnostic.

        Example:

            doc_id = ABC
            page_number = 1

        Pinecone returns:

            ABC/page1/chunk1
            ABC/page1/chunk2
            ABC/page1/chunk3

        regardless of what the document contains.
        """

        try:

            page_number = int(
                page_number
            )

        except (
            TypeError,
            ValueError,
        ):

            return []

        # --------------------------------------------------------------
        # Metadata filter
        # --------------------------------------------------------------

        metadata_filter = {
            "$and": [
                {
                    "doc_id": {
                        "$eq": doc_id
                    }
                },
                {
                    "page_number": {
                        "$eq": page_number
                    }
                },
            ]
        }

        print(
            "\n[VECTOR STORE] PAGE EXPANSION QUERY"
        )

        print(
            f"doc_id: {doc_id}"
        )

        print(
            f"page_number: {page_number}"
        )

        # --------------------------------------------------------------
        # Query Pinecone.
        #
        # We use the original query vector only as a technical vector
        # required by Pinecone. The metadata filter determines which
        # page's chunks are returned.
        # --------------------------------------------------------------

        response = _with_retry(
            lambda: self._index.query(
                vector=query_vector,
                top_k=top_k,
                filter=metadata_filter,
                include_metadata=True,
            )
        )

        results = self._matches_to_results(
            response.matches
        )

        # --------------------------------------------------------------
        # Sort by original document order.
        # --------------------------------------------------------------

        results.sort(
            key=lambda result: (
                result.chunk.chunk_index
            )
        )

        print(
            f"[VECTOR STORE] Page chunks found: "
            f"{len(results)}"
        )

        for result in results:

            print(
                f"  chunk_index="
                f"{result.chunk.chunk_index}, "
                f"chars="
                f"{len(result.chunk.text)}"
            )

        return results

    # ==================================================================
    # EXPAND RETRIEVAL RESULTS
    # ==================================================================

    def expand_results_by_page(
        self,
        results: list[RetrievalResult],
        query_vector: list[float],
        max_pages: int = 3,
    ) -> list[RetrievalResult]:
        """
        Expand relevant search results to their complete pages.

        The first retrieval identifies relevant pages.

        Then every chunk from those pages is retrieved.

        No domain-specific rules are used.
        """

        if not results:
            return []

        expanded = []

        seen_chunks = set()
        seen_pages = set()

        # --------------------------------------------------------------
        # Results are already ranked by Retriever.
        # --------------------------------------------------------------

        for result in results:

            doc_id = result.chunk.doc_id
            page_number = result.chunk.page_number

            # ----------------------------------------------------------
            # If page number is unavailable, preserve original chunk.
            # ----------------------------------------------------------

            if page_number is None:

                chunk_key = (
                    doc_id,
                    result.chunk.chunk_id,
                )

                if chunk_key not in seen_chunks:

                    expanded.append(
                        result
                    )

                    seen_chunks.add(
                        chunk_key
                    )

                continue

            page_key = (
                doc_id,
                int(page_number),
            )

            # ----------------------------------------------------------
            # Don't expand same page repeatedly.
            # ----------------------------------------------------------

            if page_key in seen_pages:
                continue

            # ----------------------------------------------------------
            # Don't expand unlimited pages.
            # ----------------------------------------------------------

            if len(seen_pages) >= max_pages:
                break

            seen_pages.add(
                page_key
            )

            # ----------------------------------------------------------
            # Fetch every chunk from this page.
            # ----------------------------------------------------------

            page_chunks = self.get_page_chunks(
                doc_id=doc_id,
                page_number=page_number,
                query_vector=query_vector,
                top_k=100,
            )

            # ----------------------------------------------------------
            # If expansion fails, preserve the original result.
            # ----------------------------------------------------------

            if not page_chunks:

                chunk_key = (
                    doc_id,
                    result.chunk.chunk_id,
                )

                if chunk_key not in seen_chunks:

                    expanded.append(
                        result
                    )

                    seen_chunks.add(
                        chunk_key
                    )

                continue

            # ----------------------------------------------------------
            # Add all page chunks.
            # ----------------------------------------------------------

            for page_chunk in page_chunks:

                chunk_key = (
                    page_chunk.chunk.doc_id,
                    page_chunk.chunk.chunk_id,
                )

                if chunk_key in seen_chunks:
                    continue

                expanded.append(
                    page_chunk
                )

                seen_chunks.add(
                    chunk_key
                )

        # --------------------------------------------------------------
        # Preserve document/page/chunk ordering.
        # --------------------------------------------------------------

        expanded.sort(
            key=lambda result: (
                result.chunk.doc_id,

                (
                    result.chunk.page_number
                    if result.chunk.page_number is not None
                    else 10**9
                ),

                result.chunk.chunk_index,
            )
        )

        return expanded

    # ==================================================================
    # DELETE DOCUMENT
    # ==================================================================

    def delete_by_doc_id(
        self,
        doc_id: str,
    ) -> None:
        """
        Delete all chunks belonging to a document.
        """

        _with_retry(
            lambda: self._index.delete(
                filter={
                    "doc_id": {
                        "$eq": doc_id
                    }
                }
            )
        )

    # ==================================================================
    # DELETE ALL
    # ==================================================================

    def delete_all(self) -> None:
        """
        Delete everything from Pinecone.
        """

        _with_retry(
            lambda: self._index.delete(
                delete_all=True
            )
        )

    # ==================================================================
    # CONVERSION HELPERS
    # ==================================================================

    @staticmethod
    def _metadata_to_result(
        vector_id: str,
        metadata,
        score: float,
    ) -> RetrievalResult:

        if metadata is None:
            metadata = {}

        page_number = metadata.get(
            "page_number"
        )

        if page_number == -1:
            page_number = None

        chunk = Chunk(
            chunk_id=vector_id,

            doc_id=metadata.get(
                "doc_id"
            ),

            text=metadata.get(
                "text",
                "",
            ),

            chunk_index=metadata.get(
                "chunk_index",
                0,
            ),

            page_number=page_number,
        )

        return RetrievalResult(
            chunk=chunk,

            score=score,

            source_filename=metadata.get(
                "source_filename",
                "Unknown source",
            ),
        )

    # ------------------------------------------------------------------

    @classmethod
    def _matches_to_results(
        cls,
        matches,
    ) -> list[RetrievalResult]:

        results = []

        for match in matches:

            metadata = getattr(
                match,
                "metadata",
                None,
            )

            if metadata is None:
                metadata = {}

            score = getattr(
                match,
                "score",
                0.0,
            )

            results.append(
                cls._metadata_to_result(
                    vector_id=match.id,
                    metadata=metadata,
                    score=score,
                )
            )

        return results