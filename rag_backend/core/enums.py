"""
Enumerations used across the RAG system.
"""

from enum import Enum


class DocumentStatus(Enum):
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    INDEXED = "indexed"
    FAILED = "failed"
    DELETED = "deleted"