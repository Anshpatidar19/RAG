"""
Central configuration for the RAG backend.
Loads settings from environment variables (via .env) so secrets
never get hardcoded into source files.
"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _get_required(key: str) -> str:
    value = os.getenv(key)
    if not value:
        raise RuntimeError(
            f"Missing required environment variable: {key}. "
            f"Add it to your .env file."
        )
    return value


def _get_optional(key: str) -> str | None:
    return os.getenv(key) or None


@dataclass(frozen=True)
class Settings:
    # Pinecone
    pinecone_api_key: str
    pinecone_index_name: str

    # Gemini
    gemini_api_key: str

    # Mistral (image OCR — optional, only needed if uploading image files)
    mistral_api_key: str | None

    # Embedding
    embedding_model: str

    # Chunking
    chunk_size: int
    chunk_overlap: int

    # Retrieval
    top_k: int
    score_threshold: float


def load_settings() -> Settings:
    return Settings(
        pinecone_api_key=_get_required("PINECONE_API_KEY"),
        pinecone_index_name=_get_required("PINECONE_INDEX_NAME"),
        gemini_api_key=_get_required("GEMINI_API_KEY"),
        mistral_api_key=_get_optional("MISTRAL_API_KEY"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
        chunk_size=int(os.getenv("CHUNK_SIZE", "800")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "100")),
        top_k=int(os.getenv("TOP_K", "3")),
        score_threshold=float(os.getenv("SCORE_THRESHOLD", "0.35")),
    )


settings = load_settings()