"""
Query-side counterpart to the vector store. Embeds the user's question,
retrieves top-k chunks, and decides whether there's enough signal to
answer at all (point 10 of the project: say when info isn't available).
"""

from config import settings
from core.models import RetrievalResult
from embedding.embedder import Embedder
from storage.vector_store import VectorStoreRepository


class Retriever:
    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStoreRepository,
        score_threshold: float | None = None,
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.score_threshold = (
            score_threshold if score_threshold is not None else settings.score_threshold
        )

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievalResult]:
        top_k = top_k or settings.top_k
        query_vector = self.embedder.embed_text(query)
        return self.vector_store.query(query_vector, top_k=top_k)

    def has_sufficient_context(self, results: list[RetrievalResult]) -> bool:
        """
        Cosine similarity scores from a normalized bge model typically
        land in a useful range around 0.3-0.8 for genuinely relevant matches.
        Below the threshold, treat it as 'not found in the knowledge base'.
        """
        return bool(results) and results[0].score >= self.score_threshold