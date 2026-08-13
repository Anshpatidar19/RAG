"""
Tool wrapper around the existing Retriever, giving agents a clean,
narrow interface for knowledge-base search. Keeps agent code from
depending on embedding/vector-store internals directly.
"""

from core.models import RetrievalResult
from retrieval.retriever import Retriever


class VectorSearchTool:
    name = "search_knowledge_base"
    description = (
        "Searches the user's uploaded documents for content relevant to a query. "
        "Returns the most similar chunks with their source filename and similarity score."
    )

    def __init__(self, retriever: Retriever):
        self.retriever = retriever

    def search(self, query: str, top_k: int | None = None) -> list[RetrievalResult]:
        return self.retriever.retrieve(query, top_k=top_k)