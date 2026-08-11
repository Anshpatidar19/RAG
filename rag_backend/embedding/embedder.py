"""
Wraps the embedding model. Converts text into vectors for storage
and querying in Pinecone.
"""

from sentence_transformers import SentenceTransformer

from config import settings


class Embedder:
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or settings.embedding_model
        self._model = SentenceTransformer(self.model_name)

    @property
    def dimension(self) -> int:
        return self._model.get_sentence_embedding_dimension()

    def embed_text(self, text: str) -> list[float]:
        """Embed a single piece of text (e.g. a user query)."""
        vector = self._model.encode(text, normalize_embeddings=True)
        return vector.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple chunks at once — more efficient than one-by-one."""
        vectors = self._model.encode(texts, normalize_embeddings=True)
        return vectors.tolist()